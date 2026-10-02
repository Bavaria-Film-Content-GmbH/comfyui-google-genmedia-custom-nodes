# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is a preview version of gemini omni custom node

from typing import Any, List, Optional, Tuple

import torch
from google import genai
from google.genai import errors as genai_errors

from PIL import Image as PIL_Image

from . import utils
from .base import VertexAIClient
from .constants import (
    GEMINI_OMNI_OUTPUT_RESOLUTION,
    GEMINI_OMNI_USER_AGENT,
    GEMINI_OMNI_VALID_ASPECT_RATIOS,
    GEMINI_OMNI_VALID_DURATION_SECONDS,
    GeminiOmniModel,
)
from .custom_exceptions import APIExecutionError, APIInputError, ConfigurationError
from .logger import get_node_logger

logger = get_node_logger(__name__)


class GeminiOmniAPI(VertexAIClient):
    """
    A client for interacting with the Google Omni API for video generation.
    """

    def __init__(
        self, project_id: Optional[str] = None, region: Optional[str] = None
    ) -> None:
        """
        Initializes the GeminiOmniAPI client.

        Args:
            project_id: The GCP project ID. If None, it will be retrieved from GCP metadata.
            region: The GCP region. If None, it will be retrieved from GCP metadata.

        Raises:
            ConfigurationError: If GCP Project or region cannot be determined or client initialization fails.
        """
        super().__init__(
            gcp_project_id=project_id, gcp_region=region, user_agent=GEMINI_OMNI_USER_AGENT
        )

    @staticmethod
    def _raise_interaction_error(
        error: Exception, previous_interaction_id: Optional[str]
    ) -> None:
        """
        Maps an interactions.create() failure to a clear APIExecutionError.

        Continuing a conversation can fail for several reasons the API doesn't
        distinguish cleanly: the referenced interaction was never stored (created
        with store=False), or its server-side availability window - undocumented,
        and observed to be short enough that continuing across separate ComfyUI runs
        can already fail even for a freshly stored interaction - has passed. This has
        been observed as a 400 "malformed format" on the interaction id in one case
        and a 400 "Precondition check failed" in another, so both are matched here to
        give an actionable error instead of a raw API error.
        """
        if previous_interaction_id and isinstance(error, genai_errors.ClientError):
            message = str(getattr(error, "message", "") or error)
            message_lower = message.lower()
            if (
                previous_interaction_id in message
                or "interaction id" in message_lower
                or "precondition" in message_lower
            ):
                raise APIExecutionError(
                    f"The conversation to continue (interaction_id={previous_interaction_id}) "
                    "is not available on the server - it may have expired, or it was never "
                    "stored because the turn that created it had 'store' disabled. Start a "
                    "new conversation instead."
                ) from error
        raise APIExecutionError(f"Gemini API Call failed: {error}") from error

    def generate_video_from_text(
        self,
        model: str,
        prompt: str,
        aspect_ratio: str,
        duration_seconds: int,
        output_resolution: str,
        previous_interaction_id: Optional[str] = None,
        store: bool = True,
    ) -> Tuple[List[str], str, str]:
        """
        Generates video from a text prompt using the Gemini Omni API.

        Args:
            model: Gemini Omni model.
            prompt: The text prompt for video generation.
            aspect_ratio: The desired aspect ratio of the video (e.g., "16:9", "1:1").
            duration_seconds: The desired duration of the video in seconds (3-10 seconds).
            output_resolution: The resolution of the generated video.
            previous_interaction_id: The interaction_id of a prior turn to continue from.
            store: Whether the interaction should be stored server-side. Required for
                `previous_interaction_id` to be usable by a later turn; when False the
                call is a faster, one-shot generation that cannot be continued.

        Returns:
            A list of file paths to the generated videos.

        Raises:
            APIInputError: If input parameters are invalid.
            APIExecutionError: If video generation fails after retries, due to API errors, or unexpected issues.
        """
        if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            raise APIInputError("Prompt cannot be empty for text-to-video generation.")
        if duration_seconds not in GEMINI_OMNI_VALID_DURATION_SECONDS:
            raise APIInputError(
                f"duration_seconds must be one of {GEMINI_OMNI_VALID_DURATION_SECONDS}, but got {duration_seconds}."
            )
        if aspect_ratio not in GEMINI_OMNI_VALID_ASPECT_RATIOS:
            raise APIInputError(
                f"Gemini Omni can only generate videos of aspect ratios {GEMINI_OMNI_VALID_ASPECT_RATIOS}. You passed aspect ratio {aspect_ratio}."
            )
        if output_resolution not in GEMINI_OMNI_OUTPUT_RESOLUTION:
            raise APIInputError(
                f"Gemini Omni can only generate videos of resolution {GEMINI_OMNI_OUTPUT_RESOLUTION}. You passed aspect ratio {output_resolution}."
            )

        model = GeminiOmniModel[model]

        # Make the API call
        try:
            logger.info(
                f"Making Gemini API call with the following Model : {model}, "
                f"previous_interaction_id={previous_interaction_id!r}, store={store}"
            )
            response_format={
                "type": "video", # optional
                "aspect_ratio": aspect_ratio,
                # "duration": str(duration_seconds),
                # "gcs_uri": ""
            }

            # if output_resolution higer than 720p:
            #     response_format["delivery"] = "uri"

            response = self.client.interactions.create(
                input=prompt,
                model=model.value,
                background=False,
                store=store,
                stream=False,
                previous_interaction_id=previous_interaction_id,
                response_format=response_format
            )
        except Exception as e:
            self._raise_interaction_error(e, previous_interaction_id)

        logger.info(f"Interaction created with id={response.id!r}")

        # Process the response
        return utils.process_video_from_interaction(
            response,
            request_params={
                "model": model.value,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "output_resolution": output_resolution,
                "duration_seconds": duration_seconds,
                "store": store,
                "previous_interaction_id": previous_interaction_id,
            },
        )


    def generate_video_from_references(
        self,
        model: str,
        prompt: str,
        image1: Optional[torch.Tensor],
        image_format: str,
        aspect_ratio: str,
        duration_seconds: int,
        output_resolution: str,
        image2: Optional[torch.Tensor],
        image3: Optional[torch.Tensor],
        video: Optional[Any] = None,
        previous_interaction_id: Optional[str] = None,
        store: bool = True,
    ) -> Tuple[List[str], str, str]:
        """
        Generates a video from the references.

        Args:
            model: Gemini Omni model.
            prompt: The text prompt for video generation.
            image1: The first reference image as a torch.Tensor.
            image_format: The format of the input images.
            aspect_ratio: The desired aspect ratio of the video (e.g., "16:9", "1:1").
            duration_seconds: The desired duration of the video in seconds (3-10 seconds).
            output_resolution: The resolution of the generated video.
            image2: The second optional reference image.
            image3: The third optional reference image.
            video: An optional ComfyUI VIDEO input to edit or extend. Per the Gemini Omni
                API, an input video used for editing/extension must be 10 seconds or less.
            previous_interaction_id: The interaction_id of a prior turn to continue from.
            store: Whether the interaction should be stored server-side. Required for
                `previous_interaction_id` to be usable by a later turn; when False the
                call is a faster, one-shot generation that cannot be continued.

        Returns:
            A list of file paths to the generated videos.

        Raises:
            APIInputError: If input parameters are invalid.
            APIExecutionError: If video generation fails after retries, due to API errors, or unexpected issues.
        """
        if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            raise APIInputError("Prompt cannot be empty for text-to-video generation.")
        if image1 is None and video is None:
            raise APIInputError(
                "At least one reference image or a video must be provided."
            )
        if duration_seconds not in GEMINI_OMNI_VALID_DURATION_SECONDS:
            raise APIInputError(
                f"duration_seconds must be one of {GEMINI_OMNI_VALID_DURATION_SECONDS}, but got {duration_seconds}."
            )
        if aspect_ratio not in GEMINI_OMNI_VALID_ASPECT_RATIOS:
            raise APIInputError(
                f"Gemini Omni can only generate videos of aspect ratios {GEMINI_OMNI_VALID_ASPECT_RATIOS}. You passed aspect ratio {aspect_ratio}."
            )
        if output_resolution not in GEMINI_OMNI_OUTPUT_RESOLUTION:
            raise APIInputError(
                f"Gemini Omni can only generate videos of resolution {GEMINI_OMNI_OUTPUT_RESOLUTION}. You passed aspect ratio {output_resolution}."
            )

        input_image_format_upper = image_format.upper()
        mime_type: str
        if input_image_format_upper == "PNG":
            mime_type = "image/png"
        elif input_image_format_upper == "JPEG":
            mime_type = "image/jpeg"
        elif input_image_format_upper == "MP4":
            mime_type = "image/mp4"
        else:
            raise APIInputError(f"Unsupported image format: {image_format}")

        input: List[Any] = []

        for image_tensor in [image1, image2, image3]:
            if image_tensor is not None:
                imageBytes = utils.tensor_to_pil_to_base64(image_tensor, image_format)
                reference_image = {"type": "image", "data": imageBytes, "mime_type": mime_type}
                logger.info("Reference Image")
                logger.info(reference_image)

                input.append(reference_image)

        if video is not None:
            video_b64 = utils.video_input_to_base64(video)
            input.append({"type": "video", "data": video_b64, "mime_type": "video/mp4"})
            logger.info("Added video input to the request.")

        model = GeminiOmniModel[model]

        # Make the API call
        try:
            logger.info(
                f"Making Gemini API call with the following Model : {model}, "
                f"previous_interaction_id={previous_interaction_id!r}, store={store}"
            )
            response_format={
                "type": "video", # optional
                "aspect_ratio": aspect_ratio,
                # "duration": str(duration_seconds),
                # "gcs_uri": ""
            }

            # if output_resolution higer than 720p:
            #     response_format["delivery"] = "uri"

            input.append({"type": "text", "text": prompt})

            response = self.client.interactions.create(
                input=input,
                model=model.value,
                background=False,
                store=store,
                stream=False,
                previous_interaction_id=previous_interaction_id,
                response_format=response_format,
                # generation_config={
                #     "video_config": {
                #         "task": "image_to_video",
                #     }
                # }
            )
        except Exception as e:
            self._raise_interaction_error(e, previous_interaction_id)

        logger.info(f"Interaction created with id={response.id!r}")

        # Process the response
        return utils.process_video_from_interaction(
            response,
            request_params={
                "model": model.value,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "output_resolution": output_resolution,
                "duration_seconds": duration_seconds,
                "store": store,
                "previous_interaction_id": previous_interaction_id,
            },
        )
