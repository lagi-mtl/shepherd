from shepherd import Shepherd, ShepherdConfig
import cv2
from shepherd.utils.visualization import VisualizationUtils

def main():
    # Create config with default model directory
    config = ShepherdConfig()
    config.thresholds['detection'] = 0.1
    viz = VisualizationUtils()
    
    # Initialize Shepherd
    shepherd = Shepherd(config=config)
    
    # Read image
    image_path = "../images/living_room.png"
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    print(f"\nProcessing image of shape {image.shape}")
    
    # Show original image
    viz.show_image(image, "Original Image")
    
    # Process frame (will automatically estimate depth since no depth_frame provided)
    results = shepherd.process_frame(image)
    
    print(f"\nFull Pipeline Results:")
    print(f"Found {len(results)} objects:")
    for i, result in enumerate(results):
        print(f"\nObject {i+1}:")
        print(f"  Caption: {result['caption']}")
        print(f"  Class: {VisualizationUtils.YOLO_CLASSES.get(int(result['detection'].get('class_id', 0)), 'Unknown')}")
        print(f"  Confidence: {result['detection']['confidence']:.2f}")
        
    # Show visualizations
    viz.show_pipeline_step("Final Results", 
                          image,
                          detections=[r['detection'] for r in results],
                          masks=[r['mask'] for r in results],
                          depth=results[0]['depth_frame'])

if __name__ == "__main__":
    main()  