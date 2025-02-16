import pandas as pd
import os
import image2dataset

def download_images(csv_path, output_folder):
    """
    Downloads images from a CSV file using image2dataset.
    
    Args:
        csv_path (str): Path to the CSV file containing 'url' and 'label' columns.
        output_folder (str): Path to the folder where images will be stored.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    if 'url' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'url' and 'label' columns")
    
    # Save the CSV in the format required by image2dataset
    csv_processed_path = os.path.join(output_folder, "input.csv")
    df.to_csv(csv_processed_path, index=False, header=False)
    
    # Run image2dataset
    image2dataset.download(
        url_list=csv_processed_path,
        input_format="csv",
        output_folder=output_folder,
        url_col=1,  # URL is in the first column
        caption_col=0,  # Label is in the second column
        output_format="webdataset",
        processes_count=8,  # Adjust based on CPU
        thread_count=32,  # Adjust based on system capability
        image_size=256,  # Resize images to 256x256 (optional)
        resize_mode="center_crop",  # Adjust resize mode if needed
        encode_format="jpg",  # Save images as JPG
        save_additional_columns=["caption"]
    )
    
    print(f"Images downloaded and saved in {output_folder}")

if __name__ == "__main__":
    csv_file = "images_urls.csv"  # path to gather the images from
    output_dir = "/media/matt/T7/dataset"  # Update output directory as needed
    os.makedirs(output_dir, exist_ok=True)
    download_images(csv_file, output_dir)
