import cv2

from shepherd import Shepherd, ShepherdConfig
from shepherd.utils.visualization import VisualizationUtils


def main():
    # Initialize Shepherd
    config = ShepherdConfig()
    config.thresholds["detection"] = 0.1
    shepherd = Shepherd(config=config)
    viz = VisualizationUtils()

    # First, process an image to store objects
    image_path = "./images/living_room.png"
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    print("\nProcessing image to store objects...")
    # This will automatically store objects in the database
    results = shepherd.process_frame(image)

    # Query examples
    queries = [
        "a couch",
        "a table",
        "a plant",
    ]

    # Run queries
    for query in queries:
        print(f"\nQuerying for: '{query}'")
        results = shepherd.database.query_objects(query, shepherd.embedder)

        if results:
            viz.show_query_results(results, query)
        else:
            print("No results found")


if __name__ == "__main__":
    main()
