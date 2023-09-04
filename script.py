from flask import Flask, render_template, request, Response, jsonify, session
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/", methods=["GET", "POST"])
def main():
    return render_template("index.html")


def generate_frames():
    try:
        video = cv2.VideoCapture(0)
        while True:
            ret, frame = video.read()
            if not ret:
                break

            bbox, label, conf = cv.detect_common_objects(frame)
            output_image = draw_bbox(frame, bbox, label, conf)

            # Convert the image to JPEG format for streaming
            ret, buffer = cv2.imencode(".jpg", output_image)
            if not ret:
                break
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    except Exception as e:
        print(str(e))
        pass


@app.route("/detect_object")
def detect_object():
    return render_template("detect_object.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    from waitress import serve

    host = "0.0.0.0"
    port = 8000
    print(f"Server running on http://{host}:{port}")
    serve(app, host=host, port=port)
