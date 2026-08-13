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

from typing import Any, Dict, List, Optional, Tuple

import torch

from .constants import GEMINI_OMNI_VALID_ASPECT_RATIOS, GeminiOmniModel
from .custom_exceptions import APIExecutionError, APIInputError, ConfigurationError
from .logger import get_node_logger
from .gemini_omni_api import GeminiOmniAPI

logger = get_node_logger(__name__)


class GeminiOmniTextToVideoNode:
    """
    A ComfyUI node for generating videos from text prompts using the Google Gemini API.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "model": (
                    [model.name for model in GeminiOmniModel],
                    {"default": GeminiOmniModel.GEMINI_OMNI_FLASH.name},
                ),
                "prompt": ("STRING", {"multiline": True}),
                "aspect_ratio": (GEMINI_OMNI_VALID_ASPECT_RATIOS, {"default": "16:9"}),
                "output_resolution": (["720p"], {"default": "720p"}),
                "duration_seconds": (
                    "INT",
                    {"default": 10, "min": 10, "max": 10, "step": 1},
                ),
            },
            "optional": {
                "interaction_id": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Use the interaction_id of a previous interaction to continue to track the conversation history and the generated video state without re-uploading the previous video",
                    }
                ),
                "gcp_project_id": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "GCP project id where Vertex AI API will query Gemini",
                    },
                ),
                "gcp_region": (
                    "STRING",
                    {
                        "default": "global",
                        "tooltip": "GCP region for Vertex AI API",
                    },
                ),
            },
        }

    RETURN_TYPES = ("VEO_VIDEO", "STRING", "STRING",)
    RETURN_NAMES = ("video_paths", "text_answer", "interaction_id")
    FUNCTION = "generate"
    CATEGORY = "Google AI/Gemini Omni"

    def generate(
        self,
        model: str = GeminiOmniModel.GEMINI_OMNI_FLASH.name,
        prompt: str = "A drone shot smoothly flies through an ancient, mist-shrouded jungle at dawn.",
        aspect_ratio: str = "16:9",
        output_resolution: str = "720p",
        duration_seconds: int = 10,
        interaction_id: Optional[str] = None,
        gcp_project_id: Optional[str] = None,
        gcp_region: Optional[str] = None,
    ) -> Tuple[List[str],str, str]:
        """
        Generates a video from a text prompt using the Google Veo 3.0 API.

        Args:
            model: Veo3 model
            prompt: The text prompt for video generation.
            aspect_ratio: The desired aspect ratio of the video.
            output_resolution: The resolution of the generated video.
            duration_seconds: The desired duration of the video in seconds.
            gcp_project_id: GCP project ID where the Veo will be queried via Vertex AI APIs
            gcp_region: GCP region for Vertex AI APIs to query Veo

        Returns:
            A tuple containing a list of file paths to the generated videos.

        Raises:
            RuntimeError: If API configuration fails, or if video generation encounters an API error.
        """
        try:
            api = GeminiOmniAPI(project_id=gcp_project_id, region=gcp_region)
        except ConfigurationError as e:
            raise RuntimeError(f"Gemini API Configuration Error: {e}") from e


        try:
            video_paths, anwswer, interaction_id = api.generate_video_from_text(
                model=model,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                output_resolution=output_resolution,
                duration_seconds=duration_seconds,
                previous_interaction_id=interaction_id,
            )
        except APIInputError as e:
            raise RuntimeError(f"Video generation configuration error: {e}") from e
        except APIExecutionError as e:
            raise RuntimeError(f"Gemini API error: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred during video generation: {e}"
            ) from e

        return (video_paths, anwswer, interaction_id)


class GeminiOmniReferenceToVideo:
    """
    A ComfyUI node for generating videos from multiple reference images
    by uploading them to GCS and using the Google Gemini API.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """
        Defines the input types for the Veo3ReferenceToVideo node.
        """
        return {
            "required": {
                "model": (
                    [model.name for model in GeminiOmniModel],
                    {"default": GeminiOmniModel.GEMINI_OMNI_FLASH.name},
                ),
                "image1": ("IMAGE",),
                "image_format": (
                    ["PNG", "JPEG"],
                    {"default": "PNG", "tooltip": "MIME type of the image"},
                ),
                "prompt": ("STRING", {"multiline": True}),
                "aspect_ratio": (GEMINI_OMNI_VALID_ASPECT_RATIOS, {"default": "16:9"}),
                "output_resolution": (["720p"], {"default": "720p"}),
                "duration_seconds": (
                    "INT",
                    {"default": 10, "min": 10, "max": 10, "step": 1},
                ),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "interaction_id": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Use the interaction_id of a previous interaction to continue to track the conversation history and the generated video state without re-uploading the previous video",
                    }
                ),
                "gcp_project_id": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "GCP project id where Vertex AI API will query Gemini",
                    },
                ),
                "gcp_region": (
                    "STRING",
                    {
                        "default": "global",
                        "tooltip": "GCP region for Vertex AI API",
                    },
                ),
            },
        }

    RETURN_TYPES = ("VEO_VIDEO", "STRING", "STRING",)
    RETURN_NAMES = ("video_paths", "text_answer", "interaction_id")
    FUNCTION = "generate_from_references"
    CATEGORY = "Google AI/Gemini Omni"

    def generate_from_references(
        self,
        model: str,
        image1: torch.Tensor,
        image_format: str,
        prompt: str,
        aspect_ratio: str,
        output_resolution: str,
        duration_seconds: int,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
        interaction_id: Optional[str] = None,
        gcp_project_id: Optional[str] = None,
        gcp_region: Optional[str] = None,
    ) -> Tuple[List[str], str, str]:
        try:
            api = GeminiOmniAPI(project_id=gcp_project_id, region=gcp_region)
        except ConfigurationError as e:
            raise RuntimeError(f"Gemini API Configuration Error: {e}") from e


        try:
            video_paths, anwswer, interaction_id = api.generate_video_from_references(
                model=model,
                prompt=prompt,
                image1=image1,
                image2=image2,
                image3=image3,
                image_format=image_format,
                aspect_ratio=aspect_ratio,
                output_resolution=output_resolution,
                duration_seconds=duration_seconds,
                previous_interaction_id=interaction_id,
            )
        except APIInputError as e:
            raise RuntimeError(f"Video generation configuration error: {e}") from e
        except APIExecutionError as e:
            raise RuntimeError(f"Gemini API error: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred during video generation: {e}"
            ) from e

        return (video_paths, anwswer, interaction_id)


NODE_CLASS_MAPPINGS = {
    "GeminiOmniTextToVideoNode": GeminiOmniTextToVideoNode,
    # "Veo3GcsUriImageToVideoNode": Veo3GcsUriImageToVideoNode,
    # "Veo3ImageToVideoNode": Veo3ImageToVideoNode,
    "GeminiOmniReferenceToVideo": GeminiOmniReferenceToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiOmniTextToVideoNode": "Gemini Omni Text To Video",
    # "Veo3GcsUriImageToVideoNode": "Veo3.1 Image To Video (GcsUriImage)",
    # "Veo3ImageToVideoNode": "Veo3.1 Image To Video",
    "GeminiOmniReferenceToVideo": "Gemini Omni Reference To Video",
}
