"""Strict contracts for pinned model sources and generated extension metadata.

The generated metadata is the shared contract between Python acquisition tooling and
the browser runtime: it records the exact graph, preprocessing, labels, and semantic
class groups needed to consume the model without duplicating constants in TypeScript.
"""

from typing import Annotated, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    NonNegativeInt,
    PositiveInt,
    field_validator,
)

Sha256 = str
Synset = str
DataType = Literal["float32"]


def _tuple_from_toml(value: object) -> object:
    """Preserve strict tuple contracts while accepting TOML's array representation."""

    return tuple(cast(list[object], value)) if isinstance(value, list) else value


class ContractModel(BaseModel):
    """Reject unknown or weakly typed contract data."""

    model_config = ConfigDict(
        alias_generator=lambda value: (
            value.split("_")[0] + "".join(part.title() for part in value.split("_")[1:])
        ),
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        strict=True,
    )


class _ModelSourceBase(ContractModel):
    """Shared immutable provenance for one model source."""

    id: str
    filename: str
    url: HttpUrl
    revision: str
    sha256: Sha256
    size_bytes: PositiveInt
    license: str
    opset: PositiveInt


class OnnxModelSource(_ModelSourceBase):
    """An upstream ONNX artifact used without conversion."""

    format: Literal["onnx"]


class TimmSafetensorsModelSource(_ModelSourceBase):
    """Pinned timm weights and the deterministic ONNX artifact they produce."""

    format: Literal["timm-safetensors"]
    architecture: str
    exporter_version: str
    artifact_sha256: Sha256
    artifact_size_bytes: PositiveInt


ModelSource = Annotated[
    OnnxModelSource | TimmSafetensorsModelSource,
    Field(discriminator="format"),
]


class LabelsSource(ContractModel):
    """Immutable provenance and integrity facts for ImageNet labels."""

    url: HttpUrl
    revision: str
    sha256: Sha256
    size_bytes: PositiveInt
    count: PositiveInt


class GraphInputSource(ContractModel):
    """Expected input graph signature."""

    name: str
    data_type: DataType
    batch_dimension: str
    channels: PositiveInt
    height: PositiveInt
    width: PositiveInt


class GraphOutputSource(ContractModel):
    """Expected output graph signature."""

    name: str
    data_type: DataType
    batch_dimension: str
    classes: PositiveInt


class GraphSource(ContractModel):
    """Expected model graph boundary."""

    input: GraphInputSource
    output: GraphOutputSource


class PreprocessingSource(ContractModel):
    """Image-to-tensor transform selected for complete source-image coverage."""

    color_space: Literal["RGB"]
    layout: Literal["NCHW"]
    resize_mode: Literal["contain"]
    allow_upscale: Literal[True]
    interpolation: Literal["bilinear"]
    padding_mode: Literal["black"]
    pixel_scale: float
    mean: tuple[float, float, float]
    standard_deviation: tuple[float, float, float]

    _convert_toml_arrays = field_validator(
        "mean",
        "standard_deviation",
        mode="before",
    )(_tuple_from_toml)


class PostprocessingSource(ContractModel):
    """Score-to-probability transform required by the graph output."""

    activation: Literal["softmax"]


class ClassesSource(ContractModel):
    """Semantic model groups resolved by synset instead of fragile indices."""

    blocked_synsets: tuple[Synset, ...]
    debug_synsets: tuple[Synset, ...]

    _convert_toml_arrays = field_validator(
        "blocked_synsets",
        "debug_synsets",
        mode="before",
    )(_tuple_from_toml)


class SourceManifest(ContractModel):
    """Complete checked-in specification for recreating the model bundle."""

    schema_version: Literal[3]
    model: ModelSource
    labels: LabelsSource
    graph: GraphSource
    preprocessing: PreprocessingSource
    postprocessing: PostprocessingSource
    classes: ClassesSource


class ModelMetadata(ContractModel):
    """Verified ONNX artifact facts exposed to runtime consumers."""

    id: str
    filename: str
    sha256: Sha256
    size_bytes: PositiveInt
    format: Literal["onnx"]
    opset: PositiveInt
    source_url: HttpUrl
    source_revision: str


class InputMetadata(ContractModel):
    """Complete browser-side preprocessing and input contract."""

    name: str
    data_type: DataType
    layout: Literal["NCHW"]
    shape: tuple[None, PositiveInt, PositiveInt, PositiveInt]
    color_space: Literal["RGB"]
    resize_mode: Literal["contain"]
    allow_upscale: Literal[True]
    interpolation: Literal["bilinear"]
    padding_mode: Literal["black"]
    pixel_scale: float
    mean: tuple[float, float, float]
    standard_deviation: tuple[float, float, float]


class LabelRecord(ContractModel):
    """One zero-indexed ImageNet output class."""

    index: NonNegativeInt
    synset: Synset
    label: str


class OutputMetadata(ContractModel):
    """Complete browser-side output and postprocessing contract."""

    name: str
    data_type: DataType
    shape: tuple[None, PositiveInt]
    activation: Literal["softmax"]
    labels: tuple[LabelRecord, ...]


class ClassGroups(ContractModel):
    """Resolved classes that drive blocking and harmless debugging."""

    blocked: tuple[LabelRecord, ...]
    debug: tuple[LabelRecord, ...]


class ArtifactMetadata(ContractModel):
    """Deterministic metadata bundled beside the model for the extension."""

    schema_version: Literal[2]
    model: ModelMetadata
    input: InputMetadata
    output: OutputMetadata
    classes: ClassGroups
