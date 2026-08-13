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

from typing import List, Optional, Tuple

import torch
from google import genai

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

    def generate_video_from_text(
        self,
        model: str,
        prompt: str,
        aspect_ratio: str,
        duration_seconds: int,
        output_resolution: str,
        previous_interaction_id: Optional[str] = None
    ) -> Tuple[List[str], str, str]:
        """
        Generates video from a text prompt using the Gemini Omni API.

        Args:
            model: Gemini Omni model.
            prompt: The text prompt for video generation.
            aspect_ratio: The desired aspect ratio of the video (e.g., "16:9", "1:1").
            duration_seconds: The desired duration of the video in seconds (3-10 seconds).
            output_resolution: The resolution of the generated video.

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
                f"Making Gemini API call with the following Model : {model}"
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
                # background=false,
                # store=false,
                # stream=false,
                # previous_interaction_id=previous_interaction_id,
                response_format=response_format
            )
        except Exception as e:
             raise APIExecutionError(f"Gemini API Call failed: {e}") from e

        # Process the response
        return utils.process_video_from_interaction(response)


    def generate_video_from_references(
        self,
        model: str,
        prompt: str,
        image1: torch.Tensor,
        image_format: str,
        aspect_ratio: str,
        duration_seconds: int,
        output_resolution: str,
        image2: Optional[torch.Tensor],
        image3: Optional[torch.Tensor],
        previous_interaction_id: Optional[str] = None
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

        Returns:
            A list of file paths to the generated videos.

        Raises:
            APIInputError: If input parameters are invalid.
            APIExecutionError: If video generation fails after retries, due to API errors, or unexpected issues.
        """
        if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            raise APIInputError("Prompt cannot be empty for text-to-video generation.")
        if image1 is None:
            raise APIInputError(
                "Image1 is required. At least reference image must be provided."
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

        model = GeminiOmniModel[model]

        # Make the API call
        try:
            logger.info(
                f"Making Gemini API call with the following Model : {model}"
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
                response_format=response_format,
                # generation_config={
                #     "video_config": {
                #         "task": "image_to_video",
                #     }
                # }
            )
        except Exception as e:
             raise APIExecutionError(f"Gemini API Call failed: {e}") from e

        # Process the response
        return utils.process_video_from_interaction(response)
