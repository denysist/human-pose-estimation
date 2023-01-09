import io
import pathlib
import shutil

import aiofiles
import fastapi
from fastapi import responses
import numpy as np
from PIL import Image
import pydantic
import uvicorn
import yaml

from library import inferences, visualizations


temporary_folder = pathlib.Path("storage")
config = "configs/app.yaml"
with open(file=config, mode="r") as file:
    config = yaml.load(file, yaml.Loader)

visualization = config["visualization"]
model = inferences.TopDownPoseModel(**config["model"])
visualizer = visualizations.Visualizer(model=model)
app = fastapi.FastAPI()


def form(cls):
    cls.__signature__ = cls.__signature__.replace(
        parameters=[
            argument.replace(default=fastapi.Form(default=argument.default))
            for argument in cls.__signature__.parameters.values()
        ]
    )

    return cls


@form
class ImageForm(pydantic.BaseModel):
    box_threshold: float = 0.5
    keypoint_threshold: float = 0.3
    line_thickness: int = 1
    point_radius: int = 3
    draw_box: bool = False
    draw_skeleton: bool = False
    draw_keypoint: bool = False


@form
class VideoForm(pydantic.BaseModel):
    box_threshold: float = 0.5
    keypoint_threshold: float = 0.3
    line_thickness: int = 1
    point_radius: int = 3
    draw_box: bool = False
    draw_skeleton: bool = False
    draw_keypoint: bool = False
    fps: float | None = None


@app.get(path="/", status_code=200)
def model() -> dict[str, str]:
    return {"message": "human pose estimation"}


@app.post(path="/image", status_code=200)
async def detect_on_image(
    input: fastapi.UploadFile = fastapi.File(),
    parameters: ImageForm = fastapi.Depends(),
) -> fastapi.Response:
    file = pathlib.Path(input.filename)
    if file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        raise fastapi.HTTPException(
            status_code=400, detail="File supports only .jpg, .jpeg or .png formats"
        )

    input = await input.read()
    input = Image.open(io.BytesIO(input))
    if input.mode != "RGB":
        input = input.convert(mode="RGB")

    image, _ = visualizer.draw_on_image(
        input=np.array(input), **dict(parameters), **visualization
    )
    image = Image.fromarray(image)
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        content = buffer.getvalue()

    return fastapi.Response(
        content=content,
        headers={"content-disposition": "attachment; filename=image.png"},
        media_type="image/png",
        status_code=200,
    )


@app.post(path="/video", status_code=200)
async def detect_on_video(
    input: fastapi.UploadFile = fastapi.File(),
    parameters: VideoForm = fastapi.Depends(),
) -> fastapi.Response:
    file = pathlib.Path(input.filename)
    if file.suffix.lower() != ".mp4":
        raise fastapi.HTTPException(
            status_code=400, detail="File supports only .mp4 format"
        )

    temporary_folder.mkdir(exist_ok=True)
    input_file = temporary_folder / "input.mp4"
    output_file = temporary_folder / "output.mp4"
    async with aiofiles.open(file=input_file, mode="wb") as file:
        input = await input.read()
        await file.write(input)

    visualizer.draw_on_video(
        input=input_file, output_file=output_file, **dict(parameters), **visualization
    )

    return responses.FileResponse(
        path=output_file,
        headers={"content-disposition": "attachment; filename=video.mp4"},
        media_type="video/mp4",
        status_code=200,
    )


if __name__ == "__main__":
    uvicorn.run(app="app:app", host="0.0.0.0", port=8000, reload=False)

    if temporary_folder.exists():
        shutil.rmtree(temporary_folder)
