'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { vaccineResourceApi, VaccineResourceApiClient } from '@/lib/api/vaccine-resource';
import { FileUpload } from '@/components/ui/file-upload';
import { VaccineChat } from '@/components/ui/vaccine-chat';
import { Upload, MessageCircle, FileText, Plus, RefreshCw, Trash2 } from 'lucide-react';

// --- Categories ---
const categories = [
  {
    key: 'clinical',
    icon: '🏥',
    name: 'Clinical Guidelines',
    description: 'How vaccines should be administered, stored, and handled. Includes timing, spacing, contraindications, etc.'
  },
  {
    key: 'safety',
    icon: '🔬',
    name: 'Safety & Efficacy Evidence',
    description: 'Real-world studies, lab results, or reports showing safety, side effects, and vaccine performance (efficacy/effectiveness).'
  },
  {
    key: 'policy',
    icon: '📋',
    name: 'Policy & Regulatory Updates',
    description: 'Recommendations or approvals from health authorities (e.g. CDC, WHO, UK Parliament). Covers who should get what, and when.'
  },
  {
    key: 'public',
    icon: '📣',
    name: 'Public Education & Messaging',
    description: 'Content designed to explain vaccines to the general public, debunk myths, or improve acceptance.'
  }
];

// --- Documents ---
const documents = [
  {
    name: 'General Best Practice Guidelines for Immunization',
    file: '/documents/general-recs.pdf',
    source: 'CDC',
    categories: ['clinical', 'safety']
  },
  {
    name: 'Green Book Chapter 14a: COVID-19 (SARS-CoV-2)',
    file: '/documents/GreenBook-chapter-14a-COVID-19-17_3_25.pdf',
    source: 'UK Health Security Agency',
    categories: ['clinical', 'policy', 'safety']
  },
  {
    name: 'WHO Position Paper: Vaccines Against Influenza (May 2022)',
    file: '/documents/WER9719-eng-fre.pdf',
    source: 'WHO',
    categories: ['policy', 'safety']
  },
  {
    name: 'CDC MMWR: COVID-19 Vaccine Recommendations 2024–2025',
    file: '/documents/mm7337e2-H.pdf',
    source: 'CDC',
    categories: ['policy', 'safety']
  },
  {
    name: 'Childhood Immunisation Statistics – UK (May 2025)',
    file: '/documents/CBP-8556.pdf',
    source: 'UK Parliament Research Briefing',
    categories: ['policy']
  },
  {
    name: 'Myths and Facts about COVID-19 Vaccines',
    file: '/documents/Myths and Facts about COVID-19 Vaccines _ CDC Archive.pdf',
    source: 'CDC',
    categories: ['public']
  },
  {
    name: 'WHO Position Paper Development Process',
    file: '/documents/position-paper-process.pdf',
    source: 'WHO',
    categories: ['policy']
  },
  {
    name: 'Why vaccination is important – NHS',
    file: '/documents/Why vaccination is important - NHS.pdf',
    source: 'NHS (UK)',
    categories: ['public']
  }
];

export function VaccineResourcesPage() {
  // --- State Management ---
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState<'browse' | 'upload' | 'chat'>('browse');
  
  // Get supported file types
  const { data: supportedTypesData } = useQuery({
    queryKey: ['vaccine-supported-types'],
    queryFn: async () => {
      const response = await vaccineResourceApi.getSupportedTypes();
      if (!response.success) throw new Error(response.error);
      return response.data!;
    }
  });

  // Get uploaded documents
  const { data: uploadedDocumentsData, refetch: refetchUploadedDocuments } = useQuery({
    queryKey: ['vaccine-uploaded-documents'],
    queryFn: async () => {
      const response = await vaccineResourceApi.getUploadedDocuments();
      if (!response.success) throw new Error(response.error);
      return response.data!;
    },
    refetchInterval: 30000 // Refetch every 30 seconds to show new uploads
  });

  // Chat functionality
  const handleSendMessage = async (message: string): Promise<string> => {
    const response = await vaccineResourceApi.chat(message);
    if (!response.success) throw new Error(response.error || 'Failed to send message');
    return response.data!.response;
  };

  // File upload functionality
  const handleFileUpload = async (file: File, metadata?: any): Promise<void> => {
    const response = await vaccineResourceApi.uploadFile(file, metadata);
    if (!response.success) throw new Error(response.error || 'Failed to upload file');
    // Refresh the uploaded documents list after successful upload
    await refetchUploadedDocuments();
  };

  // Delete document functionality
  const handleDeleteDocument = async (docId: string): Promise<void> => {
    if (!confirm('Are you sure you want to delete this document? This action cannot be undone.')) {
      return;
    }
    
    const response = await vaccineResourceApi.deleteDocument(docId);
    if (!response.success) throw new Error(response.error || 'Failed to delete document');
    
    // Refresh the uploaded documents list after successful deletion
    await refetchUploadedDocuments();
  };

  // Combine predefined and uploaded documents
  const allDocuments = [
    ...documents.map(doc => ({
      ...doc,
      isUploaded: false,
      doc_id: doc.file,
      upload_timestamp: '',
      total_chunks: 0,
      file_size: 0,
      file_type: 'pdf'
    })),
    ...(uploadedDocumentsData?.documents || []).map(doc => ({
      name: doc.title,
      file: `#uploaded-${doc.doc_id}`,
      source: doc.source,
      categories: doc.categories,
      isUploaded: true,
      doc_id: doc.doc_id,
      upload_timestamp: doc.upload_timestamp,
      total_chunks: doc.total_chunks,
      file_size: doc.file_size,
      file_type: doc.file_type
    }))
  ];

  // Filter documents based on search query
  const filteredDocuments = allDocuments.filter(doc => {
    if (!searchQuery) return true;
    
    const query = searchQuery.toLowerCase();
    const matchesName = doc.name.toLowerCase().includes(query);
    const matchesSource = doc.source.toLowerCase().includes(query);
    const matchesCategory = doc.categories.some(cat => 
      categories.find(c => c.key === cat)?.name.toLowerCase().includes(query)
    );
    
    return matchesName || matchesSource || matchesCategory;
  });

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">Vaccine Resources</h1>
        <p className="text-lg text-gray-600 max-w-3xl mx-auto">
          Access official vaccine guidelines, research, and policy documents. Upload new documents or chat with our AI assistant for personalized information.
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex justify-center mb-8">
        <div className="flex space-x-1 bg-gray-100 rounded-lg p-1">
          <button
            onClick={() => setActiveTab('browse')}
            className={`flex items-center px-4 py-2 rounded-md font-medium text-sm transition-colors ${
              activeTab === 'browse'
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <FileText className="h-4 w-4 mr-2" />
            Browse Documents
          </button>
          <button
            onClick={() => setActiveTab('upload')}
            className={`flex items-center px-4 py-2 rounded-md font-medium text-sm transition-colors ${
              activeTab === 'upload'
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <Upload className="h-4 w-4 mr-2" />
            Upload Document
          </button>
          <button
            onClick={() => setActiveTab('chat')}
            className={`flex items-center px-4 py-2 rounded-md font-medium text-sm transition-colors ${
              activeTab === 'chat'
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <MessageCircle className="h-4 w-4 mr-2" />
            AI Assistant
          </button>
        </div>
      </div>

      {/* Tab Content */}
      {activeTab === 'browse' && (
        <div>
          {/* Search Bar */}
          <div className="mb-8">
            <div className="max-w-md mx-auto flex space-x-2">
              <input
                type="text"
                placeholder="Search documents by name, source, or category..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button
                onClick={() => refetchUploadedDocuments()}
                className="px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                title="Refresh uploaded documents"
              >
                <RefreshCw className="h-4 w-4" />
              </button>
            </div>
          </div>

          {/* Category Filters */}
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4 text-center">Browse by Category</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {categories.map((cat) => (
                <div
                  key={cat.key}
                  className={`text-center p-4 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer transition-all ${
                    searchQuery === cat.name ? 'ring-2 ring-blue-400 bg-blue-50' : ''
                  }`}
                  onClick={() => setSearchQuery(cat.name)}
                >
                  <div className="text-3xl mb-2">{cat.icon}</div>
                  <div className="font-medium">{cat.name}</div>
                  <div className="text-sm text-gray-500 mt-1">{cat.description}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Documents Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredDocuments.map((doc, index) => (
              <div key={index} className={`bg-white border rounded-lg p-6 hover:shadow-md transition-shadow ${
                doc.isUploaded ? 'border-green-200 bg-green-50' : 'border-gray-200'
              }`}>
                <div className="flex items-start justify-between mb-2">
                  <h3 className="font-semibold text-gray-900">{doc.name}</h3>
                  {doc.isUploaded && (
                    <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                      📤 Uploaded
                    </span>
                  )}
                </div>
                <p className="text-sm text-gray-600 mb-2">Source: {doc.source}</p>
                {doc.isUploaded && (
                  <div className="text-xs text-gray-500 mb-3">
                    <p>📄 {doc.file_type.toUpperCase()} • {VaccineResourceApiClient.formatFileSize(doc.file_size)}</p>
                    <p>📊 {doc.total_chunks} chunks • Uploaded {new Date(doc.upload_timestamp).toLocaleDateString()}</p>
                  </div>
                )}
                <div className="flex flex-wrap gap-1 mb-4">
                  {doc.categories.map((catKey) => {
                    const category = categories.find(c => c.key === catKey);
                    return (
                      <span
                        key={catKey}
                        className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                          catKey === 'user_upload' 
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-blue-100 text-blue-800'
                        }`}
                      >
                        {category?.icon || '📄'} {category?.name || 'User Upload'}
                      </span>
                    );
                  })}
                </div>
                {doc.isUploaded ? (
                  <div className="flex items-center justify-between">
                    <div className="text-sm text-gray-600">
                      <p>✅ Available for AI queries</p>
                      <p className="text-xs text-gray-500 mt-1">Ask the AI assistant about this document</p>
                    </div>
                    <button
                      onClick={() => handleDeleteDocument(doc.doc_id)}
                      className="inline-flex items-center px-2 py-1 border border-red-200 text-xs font-medium rounded-md text-red-700 bg-red-50 hover:bg-red-100 transition-colors"
                      title="Delete document"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                ) : (
                  <a
                    href={doc.file}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 transition-colors"
                  >
                    View Document
                  </a>
                )}
              </div>
            ))}
          </div>

          {filteredDocuments.length === 0 && (
            <div className="text-center py-12">
              <p className="text-gray-500">No documents found matching your search.</p>
              <button
                onClick={() => setSearchQuery('')}
                className="mt-2 text-blue-600 hover:text-blue-700 text-sm"
              >
                Clear search
              </button>
            </div>
          )}
        </div>
      )}

      {activeTab === 'upload' && (
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Upload New Document</h2>
            <p className="text-gray-600">
              Upload vaccine-related documents to expand our knowledge base. Supported formats include PDF, Word, PowerPoint, Excel, and text files.
            </p>
          </div>

          {supportedTypesData ? (
            <FileUpload
              onFileSelect={(file, metadata) => {
                console.log('File selected:', file.name, metadata);
              }}
              onUpload={handleFileUpload}
              supportedTypes={supportedTypesData.supported_types}
              maxSizeMB={supportedTypesData.max_file_size_mb}
              showMetadataForm={true}
            />
          ) : (
            <div className="text-center py-12">
              <p className="text-gray-500">Loading upload options...</p>
            </div>
          )}
        </div>
      )}

      {activeTab === 'chat' && (
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">AI Assistant</h2>
            <p className="text-gray-600">
              Ask questions about vaccines, safety, guidelines, or any information from our document database.
            </p>
          </div>

          <VaccineChat
            onSendMessage={handleSendMessage}
            placeholder="Ask about vaccines, side effects, guidelines, or specific documents..."
          />
        </div>
      )}
    </div>
  );
}

// Default export for compatibility
export default function VaccineResources() {
  return <VaccineResourcesPage />;
} 