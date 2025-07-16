'use client';

import React, { useState, useRef, useEffect } from 'react';

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
    categories: ['clinical', 'safety', 'policy']
  },
  {
    name: 'CDC MMWR: COVID-19 Vaccine Recommendations 2024–25',
    file: '/documents/mm7337e2-H.pdf',
    source: 'CDC',
    categories: ['policy', 'safety']
  },
  {
    name: 'Childhood Immunisation Statistics – UK (May 2025)',
    file: '/documents/CBP-8556.pdf',
    source: 'UK Parliament Research Briefing',
    categories: ['policy', 'safety']
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
  // --- Chat State Management ---
  const [messages, setMessages] = useState([
    { role: 'ai', text: 'Hello! I can help you find information from official vaccine documents. What would you like to know?' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom on new message
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // --- Backend endpoint (adjust as needed) ---
  const BACKEND_ENDPOINT = 'http://localhost:8006/chat';

  // --- Send message handler ---
  const sendMessage = async () => {
    if (!input.trim() || loading) return;
    const userMsg = { role: 'user', text: input };
    setMessages(msgs => [...msgs, userMsg]);
    setLoading(true);
    try {
      const res = await fetch(BACKEND_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input })
      });
      if (!res.ok) throw new Error('Server error');
      const data = await res.json();
      setMessages(msgs => [...msgs, { role: 'ai', text: data.response || 'No response from AI.' }]);
    } catch (e) {
      setMessages(msgs => [...msgs, { role: 'ai', text: 'Sorry, there was an error contacting the AI agent.' }]);
    }
    setInput('');
    setLoading(false);
  };

  // --- Handle Enter key ---
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') sendMessage();
  };

  // Helper to get category object by key
  const getCategory = (key: string) => categories.find(cat => cat.key === key);

  // --- Filtered documents for search ---
  const filteredDocuments = documents.filter(doc => {
    const q = searchQuery.trim().toLowerCase();
    if (!q) return true;
    const nameMatch = doc.name.toLowerCase().includes(q);
    const sourceMatch = doc.source.toLowerCase().includes(q);
    const categoryMatch = doc.categories.some(catKey => {
      const cat = getCategory(catKey);
      return cat && cat.name.toLowerCase().includes(q);
    });
    return nameMatch || sourceMatch || categoryMatch;
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold text-gray-900">Vaccine Resources</h2>
        <div className="text-sm text-gray-500">
          Official medical documents and AI-powered search
        </div>
      </div>

      {/* Search */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Document Search</h3>
        <div className="flex space-x-4">
          <input 
            type="text" 
            className="flex-1 border border-gray-300 rounded-md px-4 py-2"
            placeholder="Search official vaccine documents..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      {/* Document Library */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Document List */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Available Documents</h3>
          <div className="space-y-3">
            {filteredDocuments.length === 0 && (
              <div className="text-gray-500 text-center py-8">No documents match your search.</div>
            )}
            {filteredDocuments.map((doc, idx) => (
              <div key={doc.file} className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 flex items-center space-x-3">
                <div className="text-2xl">📄</div>
                <div className="flex-1">
                  <div className="font-medium">{doc.name}</div>
                  <div className="text-sm text-gray-500">{doc.source}</div>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {doc.categories.map(catKey => {
                      const cat = getCategory(catKey);
                      return cat ? (
                        <span key={cat.key} className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-700 border border-gray-200">
                          <span className="mr-1">{cat.icon}</span>{cat.name}
                        </span>
                      ) : null;
                    })}
                  </div>
                </div>
                <a
                  href={doc.file}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-800 px-3 py-1 border border-blue-200 rounded"
                >
                  View
                </a>
              </div>
            ))}
          </div>
        </div>

        {/* AI Chat */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">AI Document Assistant</h3>
          <div className="border border-gray-300 rounded-lg h-96 flex flex-col">
            {/* Chat Messages */}
            <div className="flex-1 p-4 overflow-y-auto space-y-3">
              {messages.map((msg, idx) => (
                <div key={idx} className={
                  msg.role === 'ai'
                    ? 'flex items-start space-x-3'
                    : 'flex items-start space-x-3 justify-end'
                }>
                  {msg.role === 'ai' && (
                    <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm font-bold">AI</div>
                  )}
                  <div className={
                    msg.role === 'ai'
                      ? 'flex-1 bg-gray-100 rounded-lg p-3'
                      : 'flex-1 bg-blue-500 text-white rounded-lg p-3 max-w-xs'
                  }>
                    <p className="text-sm whitespace-pre-line">{msg.text}</p>
                  </div>
                  {msg.role === 'user' && (
                    <div className="w-8 h-8 bg-gray-400 rounded-full flex items-center justify-center text-white text-sm font-bold">U</div>
                  )}
                </div>
              ))}
              {loading && (
                <div className="flex items-start space-x-3">
                  <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm font-bold">AI</div>
                  <div className="flex-1 bg-gray-100 rounded-lg p-3">
                    <p className="text-sm animate-pulse">Thinking...</p>
                  </div>
                </div>
              )}
              <div ref={chatBottomRef} />
            </div>
            {/* Input */}
            <div className="border-t p-4">
              <div className="flex space-x-2">
                <input
                  type="text"
                  className="flex-1 border border-gray-300 rounded-md px-3 py-2 text-sm"
                  placeholder="Ask about vaccine guidelines..."
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  disabled={loading}
                />
                <button
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 text-sm disabled:opacity-50"
                  onClick={sendMessage}
                  disabled={loading || !input.trim()}
                >
                  {loading ? 'Sending...' : 'Send'}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Document Categories */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Document Categories</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {categories.map(cat => (
            <div
              key={cat.key}
              className={`text-center p-4 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer transition-all ${searchQuery === cat.name ? 'ring-2 ring-blue-400 bg-blue-50' : ''}`}
              onClick={() => setSearchQuery(cat.name)}
            >
              <div className="text-3xl mb-2">{cat.icon}</div>
              <div className="font-medium">{cat.name}</div>
              <div className="text-sm text-gray-500 mt-1">{cat.description}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
} 