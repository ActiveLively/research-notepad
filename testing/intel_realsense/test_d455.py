import cv2

def stream_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"❌ Failed to open camera index {camera_index}")
        return

    print(f"🎥 Streaming from camera index {camera_index} (Press 'q' to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame capture failed, exiting...")
            break

        cv2.imshow(f"Camera {camera_index}", frame)

        # Quit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_camera(camera_index=2)  # try 0, 1, or 2
