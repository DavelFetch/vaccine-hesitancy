'use client';

import React, { useState, useRef, useCallback } from 'react';
import { Upload, X, File, AlertCircle, CheckCircle, Loader } from 'lucide-react';

interface FileUploadProps {
  onFileSelect: (file: File, metadata?: any) => void;
  onUpload: (file: File, metadata?: any) => Promise<void>;
  supportedTypes: string[];
  maxSizeMB: number;
  disabled?: boolean;
  multiple?: boolean;
  showMetadataForm?: boolean;
}

interface FileWithMetadata extends File {
  metadata?: {
    source?: string;
    source_type?: string;
    title?: string;
    publication_date?: string;
  };
}

export function FileUpload({
  onFileSelect,
  onUpload,
  supportedTypes,
  maxSizeMB,
  disabled = false,
  multiple = false,
  showMetadataForm = true
}: FileUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<FileWithMetadata | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [uploadMessage, setUploadMessage] = useState('');
  const [metadata, setMetadata] = useState({
    source: '',
    source_type: 'document',
    title: '',
    publication_date: ''
  });

  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = (file: File): { valid: boolean; error?: string } => {
    // Check file size
    const maxSizeBytes = maxSizeMB * 1024 * 1024;
    if (file.size > maxSizeBytes) {
      return {
        valid: false,
        error: `File size (${(file.size / 1024 / 1024).toFixed(2)}MB) exceeds maximum allowed size of ${maxSizeMB}MB`
      };
    }

    // Check file type
    const fileName = file.name.toLowerCase();
    const fileExtension = fileName.substring(fileName.lastIndexOf('.'));
    
    if (!supportedTypes.includes(fileExtension)) {
      return {
        valid: false,
        error: `File type ${fileExtension} is not supported. Supported types: ${supportedTypes.join(', ')}`
      };
    }

    return { valid: true };
  };

  const handleFileSelect = (file: File) => {
    const validation = validateFile(file);
    if (!validation.valid) {
      setUploadStatus('error');
      setUploadMessage(validation.error!);
      return;
    }

    setSelectedFile(file);
    setUploadStatus('idle');
    setUploadMessage('');
    
    // Auto-populate title from filename if empty
    if (!metadata.title) {
      const nameWithoutExt = file.name.substring(0, file.name.lastIndexOf('.'));
      setMetadata(prev => ({ ...prev, title: nameWithoutExt }));
    }

    onFileSelect(file, metadata);
  };

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (disabled) return;

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelect(files[0]); // Only take first file for now
    }
  }, [disabled]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (disabled) return;

    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setUploadStatus('idle');

    try {
      await onUpload(selectedFile, metadata);
      setUploadStatus('success');
      setUploadMessage('File uploaded and processed successfully!');
      setSelectedFile(null);
      setMetadata({ source: '', source_type: 'document', title: '', publication_date: '' });
      
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (error: any) {
      setUploadStatus('error');
      setUploadMessage(error.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* File Drop Zone */}
      <div
        className={`
          relative border-2 border-dashed rounded-lg p-8 text-center transition-colors
          ${dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'}
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:border-gray-400'}
        `}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => !disabled && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple={multiple}
          accept={supportedTypes.join(',')}
          onChange={handleChange}
          className="hidden"
          disabled={disabled}
        />
        
        <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        <p className="text-lg font-medium text-gray-900 mb-2">
          {dragActive ? 'Drop file here' : 'Upload document'}
        </p>
        <p className="text-sm text-gray-500 mb-4">
          Drag and drop your file here, or click to browse
        </p>
        <p className="text-xs text-gray-400">
          Supported formats: {supportedTypes.join(', ')} • Max size: {maxSizeMB}MB
        </p>
      </div>

      {/* Selected File Display */}
      {selectedFile && (
        <div className="mt-6 p-4 border rounded-lg bg-gray-50">
          <div className="flex items-start justify-between">
            <div className="flex items-start space-x-3">
              <File className="h-6 w-6 text-blue-500 mt-1" />
              <div>
                <p className="font-medium text-gray-900">{selectedFile.name}</p>
                <p className="text-sm text-gray-500">{formatFileSize(selectedFile.size)}</p>
              </div>
            </div>
            <button
              onClick={() => {
                setSelectedFile(null);
                setUploadStatus('idle');
                setUploadMessage('');
                if (fileInputRef.current) fileInputRef.current.value = '';
              }}
              className="text-gray-400 hover:text-gray-600"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* Metadata Form */}
          {showMetadataForm && (
            <div className="mt-4 space-y-3">
              <h4 className="font-medium text-gray-900">Document Information (Optional)</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Title
                  </label>
                  <input
                    type="text"
                    value={metadata.title}
                    onChange={(e) => setMetadata(prev => ({ ...prev, title: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Document title"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Source
                  </label>
                  <input
                    type="text"
                    value={metadata.source}
                    onChange={(e) => setMetadata(prev => ({ ...prev, source: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="e.g., CDC, WHO, NHS"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Source Type
                  </label>
                  <select
                    value={metadata.source_type}
                    onChange={(e) => setMetadata(prev => ({ ...prev, source_type: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="document">Document</option>
                    <option value="guideline">Guideline</option>
                    <option value="research">Research</option>
                    <option value="policy">Policy</option>
                    <option value="report">Report</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Publication Date
                  </label>
                  <input
                    type="date"
                    value={metadata.publication_date}
                    onChange={(e) => setMetadata(prev => ({ ...prev, publication_date: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Upload Button */}
          <div className="mt-4">
            <button
              onClick={handleUpload}
              disabled={uploading || disabled}
              className={`
                w-full py-2 px-4 rounded-md font-medium text-sm transition-colors
                ${uploading || disabled
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-700'
                }
              `}
            >
              {uploading ? (
                <span className="flex items-center justify-center">
                  <Loader className="animate-spin h-4 w-4 mr-2" />
                  Processing...
                </span>
              ) : (
                'Upload & Process Document'
              )}
            </button>
          </div>
        </div>
      )}

      {/* Status Messages */}
      {uploadMessage && (
        <div className={`mt-4 p-3 rounded-md flex items-center space-x-2 ${
          uploadStatus === 'success' ? 'bg-green-50 border border-green-200' :
          uploadStatus === 'error' ? 'bg-red-50 border border-red-200' :
          'bg-blue-50 border border-blue-200'
        }`}>
          {uploadStatus === 'success' && <CheckCircle className="h-5 w-5 text-green-500" />}
          {uploadStatus === 'error' && <AlertCircle className="h-5 w-5 text-red-500" />}
          <p className={`text-sm ${
            uploadStatus === 'success' ? 'text-green-700' :
            uploadStatus === 'error' ? 'text-red-700' :
            'text-blue-700'
          }`}>
            {uploadMessage}
          </p>
        </div>
      )}
    </div>
  );
} 