import { BaseApiClient } from './base';
import { config } from '@/lib/config';
import { ApiResponse } from '@/types';

// Request/Response types for vaccine resource API
export interface VaccineResourceChatRequest {
  message: string;
}

export interface VaccineResourceChatResponse {
  response: string;
}

export interface FileUploadRequest {
  filename: string;
  file_data: string; // base64 encoded
  content_type: string;
  metadata?: {
    source?: string;
    source_type?: string;
    title?: string;
    publication_date?: string;
    [key: string]: any;
  };
}

export interface FileUploadResponse {
  success: boolean;
  message: string;
  doc_id?: string;
  filename?: string;
  chunks_count?: number;
  file_size?: number;
  file_type?: string;
  upload_timestamp?: string;
  error?: string;
}

export interface SupportedTypesResponse {
  supported_types: string[];
  max_file_size_mb: number;
}

export interface UploadedDocument {
  doc_id: string;
  filename: string;
  title: string;
  source: string;
  source_type: string;
  file_type: string;
  file_size: number;
  upload_timestamp: string;
  total_chunks: number;
  categories: string[];
}

export interface UploadedDocumentsResponse {
  documents: UploadedDocument[];
  total_count: number;
}

export interface DeleteDocumentRequest {
  doc_id: string;
}

export interface DeleteDocumentResponse {
  success: boolean;
  message: string;
  doc_id: string;
  error?: string;
}

export class VaccineResourceApiClient extends BaseApiClient {
  constructor() {
    super(config.agents.vaccineResource);
  }

  /**
   * Send a chat message to the vaccine resource agent
   */
  async chat(message: string): Promise<ApiResponse<VaccineResourceChatResponse>> {
    return this.post<VaccineResourceChatResponse>('/chat', { message }, { timeout: 60000 });
  }

  /**
   * Upload a file to be processed and stored in the knowledge base
   */
  async uploadFile(file: File, metadata?: FileUploadRequest['metadata']): Promise<ApiResponse<FileUploadResponse>> {
    try {
      // Convert file to base64
      const base64Data = await this.fileToBase64(file);
      
      const uploadRequest: FileUploadRequest = {
        filename: file.name,
        file_data: base64Data,
        content_type: file.type,
        metadata
      };

      return this.post<FileUploadResponse>('/upload', uploadRequest, { timeout: 300000 }); // 5 minute timeout for uploads
    } catch (error: any) {
      return {
        success: false,
        error: error.message || 'Failed to process file upload'
      };
    }
  }

  /**
   * Get supported file types for upload
   */
  async getSupportedTypes(): Promise<ApiResponse<SupportedTypesResponse>> {
    return this.get<SupportedTypesResponse>('/supported-types');
  }

  /**
   * Get uploaded documents from Qdrant
   */
  async getUploadedDocuments(): Promise<ApiResponse<UploadedDocumentsResponse>> {
    return this.get<UploadedDocumentsResponse>('/uploaded-documents');
  }

  /**
   * Delete an uploaded document
   */
  async deleteDocument(docId: string): Promise<ApiResponse<DeleteDocumentResponse>> {
    return this.post<DeleteDocumentResponse>('/delete-document', { doc_id: docId });
  }

  /**
   * Convert file to base64 string
   */
  private fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        if (typeof reader.result === 'string') {
          // Remove the data URL prefix (e.g., "data:application/pdf;base64,")
          const base64Data = reader.result.split(',')[1];
          resolve(base64Data);
        } else {
          reject(new Error('Failed to read file as base64'));
        }
      };
      reader.onerror = () => reject(new Error('Failed to read file'));
    });
  }

  /**
   * Validate file before upload
   */
  validateFile(file: File, supportedTypes: string[], maxSizeMB: number): { valid: boolean; error?: string } {
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
  }

  /**
   * Format file size for display
   */
  static formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
}

// Export singleton instance
export const vaccineResourceApi = new VaccineResourceApiClient(); 