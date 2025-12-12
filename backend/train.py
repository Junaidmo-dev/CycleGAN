from ultralytics import YOLO
import os

def train_model():
    # Load a model
    # Using yolov8n.pt (nano) for fastest training on small dataset
    # We could use yolov8s.pt or larger if accuracy is poor
    model = YOLO("yolov8n.pt") 

    # Train the model
    # Point to the absolute path of data.yaml to avoid confusion
    yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "dataset", "data.yaml"))
    
    print(f"Starting training with config: {yaml_path}")
    
    try:
        results = model.train(
            data=yaml_path, 
            epochs=50, 
            imgsz=640, 
            batch=2, 
            name="custom_underwater_yolo"
        )
        print("Training complete.")
        print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_model()
