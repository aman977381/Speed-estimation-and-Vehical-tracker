import argparse

import cv2
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, deque
import numpy as np

SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

Target_width = 25
Target_height = 250
TARGET = np.array(
    [
        [0,0],
        [Target_width-1,0],
        [Target_width-1,Target_height-1],
        [0,Target_height-1]
    ]
)

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)
    

def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Ultralytics"
    )

    parser.add_argument(
        "--source_video_path", ## data\vehicles.mp4
        required=True, 
        help = "Path to the source video file",
        type = str
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argument()

    model = YOLO("yolov8x.pt")

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)
    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    bounding_box = sv.BoundingBoxAnnotator(
        thickness=thickness,
        color_lookup=sv.ColorLookup.TRACK
        )
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.TRACK
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps*2,
        position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.TRACK
    )



    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    polygon_zone = sv.PolygonZone(polygon=SOURCE,frame_resolution_wh=video_info.resolution_wh)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)


    # Set up video writer
    output_path = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
    out = cv2.VideoWriter(output_path, fourcc, video_info.fps, (video_info.resolution_wh))

    coordinates = defaultdict(lambda:deque(maxlen=video_info.fps))


    for frame in frame_generator:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_track.update_with_detections(detections=detections)

        points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
        points = view_transformer.transform_points(points=points).astype(int)

        labels = []
        for tracker_id,[_,y] in zip(detections.tracker_id,points):
            coordinates[tracker_id].append(y)
            if len(coordinates[tracker_id])<video_info.fps/2:
                labels.append(f'#{tracker_id}')
            else:
                coordinates_start = coordinates[tracker_id][-1]
                coordinates_end = coordinates[tracker_id][0]
                distance = abs(coordinates_start-coordinates_end)
                time = len(coordinates[tracker_id])/video_info.fps
                speed = distance/time * 3.6
                labels.append(f'#{tracker_id} {int(speed)} km/h')

        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame,detections=detections
        )
        annotated_frame = bounding_box.annotate(
            scene=annotated_frame,detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,detections=detections, labels=labels
        )

        out.write(annotated_frame)

        screen_res = 1280, 720  # example screen resolution, adjust as needed
        scale_width = screen_res[0] / annotated_frame.shape[1]
        scale_height = screen_res[1] / annotated_frame.shape[0]
        scale = min(scale_width, scale_height)

        # The window will be resized to fit the screen resolution while maintaining the aspect ratio
        window_width = int(annotated_frame.shape[1] * scale)
        window_height = int(annotated_frame.shape[0] * scale)

        # Resize the image
        resized_image = cv2.resize(annotated_frame, (window_width, window_height))


        cv2.imshow("annotated_frame", resized_image)
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()