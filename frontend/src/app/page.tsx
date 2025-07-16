'use client';

import { useState } from 'react';
import { HealthBoard } from '@/components/pages/health-board';
import { SocialMediaPage } from '@/components/pages/social-media';
import { VaccineResourcesPage } from '@/components/pages/vaccine-resources';
import { VoiceAnalysisPage } from '@/components/pages/voice-analysis';
import { PageType } from '@/types';

const tabs = [
  { id: 'health-board' as PageType, label: 'Health Board', icon: '🗺️' },
  { id: 'social-media' as PageType, label: 'Social Media Analysis', icon: '📱' },
  { id: 'vaccine-resources' as PageType, label: 'Vaccine Resources', icon: '📚' },
  { id: 'voice-analysis' as PageType, label: 'Voice Analysis', icon: '🎤' },
];

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<PageType>('health-board');

  const renderPage = () => {
    switch (activeTab) {
      case 'health-board':
        return <HealthBoard />;
      case 'social-media':
        return <SocialMediaPage />;
      case 'vaccine-resources':
        return <VaccineResourcesPage />;
      case 'voice-analysis':
        return <VoiceAnalysisPage />;
      default:
        return <HealthBoard />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Vaccine Hesitancy Analysis Platform
              </h1>
              <p className="text-sm text-gray-600 mt-1">
                Comprehensive insights for health affairs professionals
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-500">
                Health Affairs Dashboard
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <nav className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <span className="text-lg">{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Page Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderPage()}
      </main>
    </div>
  );
}
