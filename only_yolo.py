import torch
import cv2
from ultralytics import YOLO
from collections import defaultdict
import time
import pickle
import queue
import os
import threading

from typing import Union

def create_list_dict():
    return []

class FIFOStream:
    def __init__(self, maxsize=0):
        self.queue = queue.Queue(maxsize)

    def write(self, chunk: Union[bytes, None]):
        if chunk:
            print(f"Queued {len(chunk)} bytes")
        self.queue.put(chunk)

    def read(self):
        chunk = self.queue.get(True)
        if chunk is None:  # EOF marker encountered
            raise EOFError()
        return chunk


def do_pickling(fifo: FIFOStream, obj):
    pickle.dump(obj, fifo, protocol=pickle.HIGHEST_PROTOCOL)
    fifo.write(None)  # write EOF marker after Pickle is done


def upload_from_file(fifo):
    n = 0
    while True:
        try:
            chunk = fifo.read()
        except EOFError:
            break
        n += len(chunk)
        print(f"Uploading chunk of size {len(chunk)}")
    print(f"Finished uploading {n} bytes!")






# data = {a: os.urandom(1024) for a in range(500)}

show = True
past_frame_len = 20
input = "input_videos/amsterdam-short.mp4"
output = "output_videos/amsterdam-out.mp4"
cap = cv2.VideoCapture(input)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec (here, MP4)
video_writer = cv2.VideoWriter(output, fourcc, cap.get(cv2.CAP_PROP_FPS),
                                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# t = np.arange(self.past_frame_len, dtype=np.uint8).reshape(-1, 1)
# t_future = np.arange(self.past_frame_len, self.past_frame_len + self.prediction_frame_len).reshape(-1, 1)
# self.loop()
device = "cuda" if torch.cuda.is_available() else "mps"

model = YOLO('yolov8l.pt')  # pretrained YOLOv8n model

model = model.to(device)
track_history = defaultdict(create_list_dict)

# fifo = FIFOStream(maxsize=0)  # adjust maxsize to something larger in real use :)
# dumper = threading.Thread(target=do_pickling, args=(fifo, data))
# dumper.start()
# upload_from_file(fifo)
# dumper.join()


time_sum = 0
frames = 0
while cap.isOpened():
    t = time.time()
    success, frame = cap.read()
    if success:
        frames += 1
        results = model.track(frame, persist=True, verbose=False)
        boxes = results[0].boxes.xywh
        track_ids = results[0].boxes.id
        name_ids = results[0].boxes.cls
        curr_annotated_frame = results[0].plot()
        if track_ids != None:
            for box, track_id, name_id in zip(boxes, track_ids, name_ids):
                track_id = int(track_id)
                x, y, w, h = box
                name = results[0].names[int(name_id)]
                track = track_history[(track_id, name)]
                track.append(box[:2])
                if show:
                    points = track[-past_frame_len:]
                    draw = False
                    points = torch.stack(points)
                    points = points.unsqueeze(1)
                    points = points.to(torch.int).numpy()
                    cv2.circle(curr_annotated_frame, (int(x), int(y)), radius=0, color=(0, 0, 255), thickness=15)
                    cv2.polylines(curr_annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
        time_sum += time.time() - t
        t = time.time()
        if show:
            cv2.imshow("YOLOv8 Tracking", curr_annotated_frame)
        video_writer.write(curr_annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
    # frame_count += 1

print(track_history)
print("Time avg: ", time_sum / frames)
with open('track_history_old.pkl', 'wb') as f:
    pickle.dump(track_history, f)
# with open('../track_predictions_old.pkl', 'wb') as f:
#     pickle.dump(self.track_predictions, f)

video_writer.release()
cap.release()
cv2.destroyAllWindows()