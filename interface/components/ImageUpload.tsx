"use client";

import React from "react";
import { Button } from "@/components/ui/button"; // Adjust the import based on your project structure.
import { UploadIcon } from "lucide-react";

interface ImageUploadProps {
  onFileChange: (file: File) => void;
}

export default function ImageUpload({ onFileChange }: ImageUploadProps) {
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileChange(e.target.files[0]);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 p-8 rounded-md hover:border-blue-500 transition-colors">
      <UploadIcon className="w-12 h-12 text-gray-500 mb-3" />
      <p className="text-gray-600 text-center mb-4">
        Drag and drop an image here, or click the button below to browse files.
      </p>
      <label htmlFor="file-upload" className="cursor-pointer">
        <Button variant="outline" className="flex items-center gap-2">
          Browse Files
          <input id="file-upload" type="file" accept="image/*" onChange={handleFileChange} className="sr-only" />
        </Button>
      </label>
    </div>
  );
}
