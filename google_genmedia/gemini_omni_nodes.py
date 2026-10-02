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

from .constants import (
    GEMINI_OMNI_CONTEXT_TYPE,
    GEMINI_OMNI_OUTPUT_RESOLUTION,
    GEMINI_OMNI_VALID_ASPECT_RATIOS,
    GeminiOmniModel,
)
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
                "output_resolution": (GEMINI_OMNI_OUTPUT_RESOLUTION, {"default": "720p"}),
                "duration_seconds": (
                    "INT",
                    {"default": 10, "min": 3, "max": 10, "step": 1},
                ),
            },
            "optional": {
                "context": (
                    GEMINI_OMNI_CONTEXT_TYPE,
                    {
                        "tooltip": "Connect the context output of a previous Gemini Omni node to continue that conversation (its history and generated video state) instead of starting a new one.",
                    }
                ),
                "store": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Keep this turn stored server-side so it can be continued later via its context output. Disable for a faster, one-shot generation that cannot be continued.",
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

    RETURN_TYPES = ("VEO_VIDEO", "STRING", GEMINI_OMNI_CONTEXT_TYPE,)
    RETURN_NAMES = ("video_paths", "text_answer", "context")
    FUNCTION = "generate"
    CATEGORY = "Google AI/Gemini Omni"

    def generate(
        self,
        model: str = GeminiOmniModel.GEMINI_OMNI_FLASH.name,
        prompt: str = "A drone shot smoothly flies through an ancient, mist-shrouded jungle at dawn.",
        aspect_ratio: str = "16:9",
        output_resolution: str = "720p",
        duration_seconds: int = 10,
        context: Optional[Dict[str, Any]] = None,
        store: bool = True,
        gcp_project_id: Optional[str] = None,
        gcp_region: Optional[str] = None,
    ) -> Tuple[List[str], str, Dict[str, Any]]:
        """
        Generates a video from a text prompt using the Google Gemini Omni API.

        Args:
            model: Gemini Omni model.
            prompt: The text prompt for video generation.
            aspect_ratio: The desired aspect ratio of the video.
            output_resolution: The resolution of the generated video.
            duration_seconds: The desired duration of the video in seconds.
            context: The context of a previous Gemini Omni turn to continue, if any.
            store: Whether to keep this turn stored server-side so it can be continued later.
            gcp_project_id: GCP project ID where Gemini will be queried via Vertex AI APIs.
            gcp_region: GCP region for Vertex AI APIs to query Gemini.

        Returns:
            A tuple containing a list of file paths to the generated videos, the text
            answer, and a context object for continuing this conversation.

        Raises:
            RuntimeError: If API configuration fails, or if video generation encounters an API error.
        """
        try:
            api = GeminiOmniAPI(project_id=gcp_project_id, region=gcp_region)
        except ConfigurationError as e:
            raise RuntimeError(f"Gemini API Configuration Error: {e}") from e

        if context and context.get("model") and context["model"] != model:
            logger.warning(
                f"Continuing a conversation started with model '{context['model']}' using "
                f"a different model '{model}'; cross-model continuation is not documented "
                "by the API and may not behave as expected."
            )

        try:
            video_paths, answer, new_interaction_id = api.generate_video_from_text(
                model=model,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                output_resolution=output_resolution,
                duration_seconds=duration_seconds,
                previous_interaction_id=context.get("interaction_id") if context else None,
                store=store,
            )
        except APIInputError as e:
            raise RuntimeError(f"Video generation configuration error: {e}") from e
        except APIExecutionError as e:
            raise RuntimeError(f"Gemini API error: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred during video generation: {e}"
            ) from e

        new_context = {
            "interaction_id": new_interaction_id,
            "model": model,
            "text_answer": answer,
            "turn": context.get("turn", 0) + 1 if context else 1,
        }
        return (video_paths, answer, new_context)


class GeminiOmniReferenceToVideo:
    """
    A ComfyUI node for generating videos from reference images and/or a
    reference video using the Google Gemini Omni API.
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
                "image_format": (
                    ["PNG", "JPEG"],
                    {"default": "PNG", "tooltip": "MIME type of the reference images"},
                ),
                "prompt": ("STRING", {"multiline": True}),
                "aspect_ratio": (GEMINI_OMNI_VALID_ASPECT_RATIOS, {"default": "16:9"}),
                "output_resolution": (GEMINI_OMNI_OUTPUT_RESOLUTION, {"default": "720p"}),
                "duration_seconds": (
                    "INT",
                    {"default": 10, "min": 3, "max": 10, "step": 1},
                ),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "video": (
                    "VIDEO",
                    {
                        "tooltip": "An optional video to edit or extend. Per the Gemini Omni API, videos used for editing/extension must be 10 seconds or less unless continuing a stored conversation via context.",
                    },
                ),
                "context": (
                    GEMINI_OMNI_CONTEXT_TYPE,
                    {
                        "tooltip": "Connect the context output of a previous Gemini Omni node to continue that conversation (its history and generated video state) instead of starting a new one.",
                    }
                ),
                "store": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Keep this turn stored server-side so it can be continued later via its context output. Disable for a faster, one-shot generation that cannot be continued.",
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

    RETURN_TYPES = ("VEO_VIDEO", "STRING", GEMINI_OMNI_CONTEXT_TYPE,)
    RETURN_NAMES = ("video_paths", "text_answer", "context")
    FUNCTION = "generate_from_references"
    CATEGORY = "Google AI/Gemini Omni"

    def generate_from_references(
        self,
        model: str,
        image_format: str,
        prompt: str,
        aspect_ratio: str,
        output_resolution: str,
        duration_seconds: int,
        image1: Optional[torch.Tensor] = None,
        image2: Optional[torch.Tensor] = None,
        image3: Optional[torch.Tensor] = None,
        video: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        store: bool = True,
        gcp_project_id: Optional[str] = None,
        gcp_region: Optional[str] = None,
    ) -> Tuple[List[str], str, Dict[str, Any]]:
        try:
            api = GeminiOmniAPI(project_id=gcp_project_id, region=gcp_region)
        except ConfigurationError as e:
            raise RuntimeError(f"Gemini API Configuration Error: {e}") from e

        if context and context.get("model") and context["model"] != model:
            logger.warning(
                f"Continuing a conversation started with model '{context['model']}' using "
                f"a different model '{model}'; cross-model continuation is not documented "
                "by the API and may not behave as expected."
            )

        try:
            video_paths, answer, new_interaction_id = api.generate_video_from_references(
                model=model,
                prompt=prompt,
                image1=image1,
                image2=image2,
                image3=image3,
                video=video,
                image_format=image_format,
                aspect_ratio=aspect_ratio,
                output_resolution=output_resolution,
                duration_seconds=duration_seconds,
                previous_interaction_id=context.get("interaction_id") if context else None,
                store=store,
            )
        except APIInputError as e:
            raise RuntimeError(f"Video generation configuration error: {e}") from e
        except APIExecutionError as e:
            raise RuntimeError(f"Gemini API error: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred during video generation: {e}"
            ) from e

        new_context = {
            "interaction_id": new_interaction_id,
            "model": model,
            "text_answer": answer,
            "turn": context.get("turn", 0) + 1 if context else 1,
        }
        return (video_paths, answer, new_context)


class GeminiOmniLoadContext:
    """
    A ComfyUI node for resuming a Gemini Omni conversation from a previously
    seen interaction_id, e.g. after restarting ComfyUI or reloading a workflow.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "interaction_id": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "A previously seen interaction_id to resume as a context you can wire into a Gemini Omni node.",
                    },
                ),
            },
            "optional": {
                "model": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional: the model the conversation was started with, for reference only.",
                    },
                ),
            },
        }

    RETURN_TYPES = (GEMINI_OMNI_CONTEXT_TYPE,)
    RETURN_NAMES = ("context",)
    FUNCTION = "load"
    CATEGORY = "Google AI/Gemini Omni"

    def load(self, interaction_id: str, model: str = "") -> Tuple[Dict[str, Any]]:
        if not interaction_id or not interaction_id.strip():
            raise RuntimeError("interaction_id cannot be empty.")

        return ({
            "interaction_id": interaction_id.strip(),
            "model": model.strip(),
            "text_answer": "",
            "turn": 0,
        },)


NODE_CLASS_MAPPINGS = {
    "GeminiOmniTextToVideoNode": GeminiOmniTextToVideoNode,
    # "Veo3GcsUriImageToVideoNode": Veo3GcsUriImageToVideoNode,
    # "Veo3ImageToVideoNode": Veo3ImageToVideoNode,
    "GeminiOmniReferenceToVideo": GeminiOmniReferenceToVideo,
    "GeminiOmniLoadContext": GeminiOmniLoadContext,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiOmniTextToVideoNode": "Gemini Omni Text To Video",
    # "Veo3GcsUriImageToVideoNode": "Veo3.1 Image To Video (GcsUriImage)",
    # "Veo3ImageToVideoNode": "Veo3.1 Image To Video",
    "GeminiOmniReferenceToVideo": "Gemini Omni Reference/Video To Video",
    "GeminiOmniLoadContext": "Gemini Omni Load Context",
}
