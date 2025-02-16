// components/ImageUpload.tsx
import React, { useState, useRef } from "react";

type ImageUploadProps = {
  onFileChange: (file: File) => void;
};

const ImageUpload: React.FC<ImageUploadProps> = ({ onFileChange }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      // Create a URL for the preview image
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      onFileChange(file);
      // Optionally clear the files from the dataTransfer
      e.dataTransfer.clearData();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      onFileChange(file);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div
      onClick={handleClick}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      style={{
        border: "2px dashed #ccc",
        padding: "20px",
        textAlign: "center",
        cursor: "pointer",
      }}
    >
      {previewUrl ? (
        <img src={previewUrl} alt="Preview" style={{ maxWidth: "100%", maxHeight: "200px", objectFit: "contain" }} />
      ) : (
        "Drag and drop an image here, or click to select."
      )}
      <input type="file" accept="image/*" ref={fileInputRef} onChange={handleFileChange} style={{ display: "none" }} />
    </div>
  );
};

export default ImageUpload;
