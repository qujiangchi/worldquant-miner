'use client';

import { useState, useCallback } from 'react';
import { motion } from 'motion/react';
import { IconUpload, IconFile, IconX, IconCheck } from '@tabler/icons-react';

interface FileUploaderProps {
  onFileUploaded: (file: File) => void;
}

export function FileUploader({ onFileUploaded }: FileUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file && file.type === 'application/pdf') {
      handleFileUpload(file);
    } else {
      alert('Please upload a PDF file');
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type === 'application/pdf') {
      handleFileUpload(file);
    } else {
      alert('Please upload a PDF file');
    }
  }, []);

  const handleFileUpload = async (file: File) => {
    setUploadStatus('uploading');
    setUploadedFile(file);
    
    try {
      // In a real implementation, this would upload the file to a server
      // const formData = new FormData();
      // formData.append('file', file);
      // const response = await fetch('/api/upload', {
      //   method: 'POST',
      //   body: formData,
      // });
      // const data = await response.json();
      
      // For now, just simulate a successful upload
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      setUploadStatus('success');
      onFileUploaded(file);
    } catch (error) {
      console.error('Error uploading file:', error);
      setUploadStatus('error');
      alert('Failed to upload file. Please try again.');
    }
  };

  const handleRemoveFile = () => {
    setUploadedFile(null);
    setUploadStatus('idle');
  };

  return (
    <motion.div
      className="backdrop-blur-md bg-white/10 p-6 rounded-xl border border-white/20"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h3 className="text-lg font-bold mb-4">Upload Research Paper</h3>
      
      {!uploadedFile ? (
        <div
          className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${
            isDragging
              ? 'border-blue-500/50 bg-blue-500/10'
              : 'border-white/20 hover:border-white/40'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="flex flex-col items-center gap-4">
            <div className={`p-3 rounded-full ${
              isDragging ? 'bg-blue-500/20 text-blue-300' : 'bg-white/10 text-blue-200'
            }`}>
              <IconUpload className="h-8 w-8" />
            </div>
            
            <div>
              <p className="text-lg font-medium">
                Drag and drop your PDF file here
              </p>
              <p className="text-sm text-blue-200 mt-1">
                or click to browse files
              </p>
            </div>
            
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileSelect}
              className="hidden"
              id="file-upload"
            />
            <label
              htmlFor="file-upload"
              className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors cursor-pointer"
            >
              Select File
            </label>
          </div>
        </div>
      ) : (
        <motion.div
          className="p-4 rounded-lg border border-white/20 bg-white/5"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className={`p-2 rounded-full ${
                uploadStatus === 'success'
                  ? 'bg-green-500/20 text-green-400'
                  : uploadStatus === 'error'
                  ? 'bg-red-500/20 text-red-400'
                  : 'bg-blue-500/20 text-blue-400'
              }`}>
                {uploadStatus === 'uploading' ? (
                  <div className="animate-spin h-5 w-5 border-2 border-blue-400 border-t-transparent rounded-full"></div>
                ) : uploadStatus === 'success' ? (
                  <IconCheck className="h-5 w-5" />
                ) : (
                  <IconX className="h-5 w-5" />
                )}
              </div>
              
              <div>
                <div className="flex items-center gap-2">
                  <IconFile className="h-5 w-5 text-blue-200" />
                  <span className="font-medium">{uploadedFile.name}</span>
                </div>
                <p className="text-sm text-blue-200 mt-1">
                  {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>
            
            <button
              onClick={handleRemoveFile}
              className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
            >
              <IconX className="h-5 w-5" />
            </button>
          </div>
          
          {uploadStatus === 'success' && (
            <motion.div
              className="mt-4 p-3 rounded-lg bg-green-500/10 border border-green-500/30 text-green-400 text-sm"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              File uploaded successfully
            </motion.div>
          )}
        </motion.div>
      )}
    </motion.div>
  );
} 