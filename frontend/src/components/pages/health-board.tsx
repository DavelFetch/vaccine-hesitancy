'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import { hesitancyInsightsApi, fetchAllRegionsData, normalizeRegionName, buildRegionValueMap } from '@/lib/api/hesitancy-insights';
import { Button, Select, Loading } from '@/components/ui';
import { VHBarChart, UKMap, DataTable } from '@/components/charts';
import { ChevronDown, RefreshCw, Info } from 'lucide-react';

type ViewType = 'map' | 'chart' | 'table';

interface VHDataItem {
  [key: string]: any;
}

export function HealthBoard() {
  const [selectedDimension, setSelectedDimension] = useState('region');
  const [selectedValue, setSelectedValue] = useState('England');
  const [viewType, setViewType] = useState<ViewType>('map');
  const [selectedSubgroup, setSelectedSubgroup] = useState<string>('');
  const [selectedBarrierGroup, setSelectedBarrierGroup] = useState<string>('');

  // Validation helper function
  const isValidDimensionValue = useCallback((dimension: string, value: string) => {
    const allDimensions = hesitancyInsightsApi.getAllDimensions();
    const availableValues = allDimensions[dimension] || [];
    return availableValues.includes(value);
  }, []);

  // Query for the selected dimension data with validation
  const { data: queryData, isLoading: isQueryLoading, error: queryError } = useQuery({
    queryKey: ['vh-data', selectedDimension, selectedValue, selectedSubgroup, selectedBarrierGroup],
    queryFn: async () => {
      // Double-check validation before API call
      if (!isValidDimensionValue(selectedDimension, selectedValue)) {
        throw new Error(`Invalid combination: ${selectedDimension} = ${selectedValue}`);
      }
      if (selectedDimension === 'age_sex') {
        return (
          await hesitancyInsightsApi.getAgeSexData(
            selectedValue,
            selectedSubgroup as "Men" | "Women" | undefined
          )
        ).data?.data || [];
      }
      if (selectedDimension === 'barriers') {
        return (
          await hesitancyInsightsApi.getBarriersData(selectedValue, selectedBarrierGroup || undefined)
        ).data?.data || [];
      }    
  
      return (
        await hesitancyInsightsApi.getVHData({
          dimension: selectedDimension,
          value: selectedValue
        })
      ).data?.data || [];
    },
    // Only enable query if we have a valid dimension-value combination
    enabled: isValidDimensionValue(selectedDimension, selectedValue),
  });

  // Query for all regions data (for map)
  const { data: mapData, isLoading: isMapLoading } = useQuery({
    queryKey: ['all-regions-map-data'],
    queryFn: fetchAllRegionsData,
    enabled: selectedDimension === 'region' && viewType === 'map',
  });

  // Get available values for the selected dimension
  const getAvailableValues = () => {
    const allDimensions = hesitancyInsightsApi.getAllDimensions();
    return allDimensions[selectedDimension] || [];
  };

  // Atomic dimension change handler to prevent race conditions
  const handleDimensionChange = useCallback((newDimension: string) => {
    const allDimensions = hesitancyInsightsApi.getAllDimensions();
    const availableValues = allDimensions[newDimension] || [];
    const defaultValue = availableValues.length > 0 ? availableValues[0] : '';
    
    // Update both values atomically to prevent invalid combinations
    setSelectedDimension(newDimension);
    setSelectedValue(defaultValue);
    setSelectedSubgroup('');
    setSelectedBarrierGroup('');
    
    // Handle view type changes
    if (newDimension === 'region') {
      setViewType('map');
    } else if (viewType === 'map') {
      setViewType('chart');
    }
  }, [viewType]);

  // Atomic state update for quick actions
  const handleQuickRegionalMap = useCallback(() => {
    setSelectedDimension('region');
    setSelectedValue('England');
    setViewType('map');
  }, []);

  // Helper to determine if we should show map
  const shouldShowMap = selectedDimension === 'region' && viewType === 'map';
  
  // Helper to get the field name for the current dimension
  const getDimensionFieldName = (dimension: string, item: any) => {
    // Map dimension to the actual field name in the data
    const fieldMapping: Record<string, string> = {
      'region': 'region',
      'sex': 'sex', 
      'ethnicity': 'ethnicity',
      'age_band': 'age_band',
      'age_group': 'age_group',
      'religion': 'religion',
      'disability_status': 'disability_status',
      'cev_status': 'cev_status',
      'health_condition': 'health_condition',
      'health_general_condition': 'health_general_condition',
      'imd_quintile': 'imd_quintile',
      'employment_status': 'employment_status',
      'expense_affordability': 'expense_affordability',
      'household_type': 'household_type',
      'caregiver_status': 'caregiver_status',
      'age_sex': 'group',
      'trends': 'period',
      'barriers': 'block',
      'reasons': 'block'
    };
    
    const fieldName = fieldMapping[dimension];
    return fieldName ? item[fieldName] : 'Unknown';
  };

  // Process data for different visualizations
  const getProcessedData = () => {
    if (shouldShowMap && mapData) {
      // For map: process all regions data to include both hesitancy and sentiment percentages
      const regionMap = new Map();
      
      // Group data by region and extract both measures
      mapData.forEach(item => {
        const region = item.region;
        if (!regionMap.has(region)) {
          regionMap.set(region, {
            region: region,
            value: 0, // Default hesitancy value for backward compatibility
            hesitancyPercent: null,
            sentimentPercent: null,
            measure: item.measure
          });
        }
        
        const regionData = regionMap.get(region);
        
        // Check for vaccine hesitancy measures
        const hesitancyMeasures = [
          'Vaccine hesitancy1,2',
          'Vaccine hesitancy2,3', 
          'Vaccine hesitancy',
          'Vaccine hesitancy1',
          'Vaccine hesitancy2'
        ];
        
        // Check for positive sentiment measures
        const sentimentMeasures = [
          'Positive vaccine sentiment1,2',
          'Positive vaccine sentiment2,3',
          'Positive vaccine sentiment',
          'Positive vaccine sentiment1',
          'Positive vaccine sentiment2'
        ];
        
        if (hesitancyMeasures.some(measure => item.measure === measure)) {
          regionData.hesitancyPercent = item.percent;
          regionData.value = item.percent; // For backward compatibility
        } else if (sentimentMeasures.some(measure => item.measure === measure)) {
          regionData.sentimentPercent = item.percent;
        }
      });
      
      return Array.from(regionMap.values()).filter(item => 
        item.hesitancyPercent !== null || item.sentimentPercent !== null
      );
    } else if (queryData) {
      // For chart and table views: show all measures for the selected dimension value
      return queryData
        .filter((item: any) => {
          const val = item.percent !== undefined ? item.percent : item.value;
          // Exclude 'Weighted count' and 'Sample size' from chart/table
          if (item.measure === 'Weighted count' || item.measure === 'Sample size') return false;
          return item.measure && 
                 item.measure !== 'N/A' && 
                 item.measure.trim() !== '' &&
                 val !== null && 
                 val !== undefined &&
                 !isNaN(val);
        })
        .map((item: any) => ({
          name: item.measure, // Use measure as the name for x-axis
          value: item.percent !== undefined ? item.percent : item.value,
          measure: item.measure,
          region: getDimensionFieldName(selectedDimension, item),
          ...item
        }));
    }
    return [];
  };

  const processedData = getProcessedData();
  const isLoading = shouldShowMap ? isMapLoading : isQueryLoading;

  // Debug logging to help with troubleshooting
  React.useEffect(() => {
    if (queryData) {
      console.log('🔍 Debug - Raw query data:', queryData);
      console.log('🔍 Debug - Selected dimension:', selectedDimension);
      console.log('🔍 Debug - Selected value:', selectedValue);
      console.log('🔍 Debug - Processed data:', processedData);
      console.log('🔍 Debug - Available measures:', [...new Set(queryData.map((item: any) => item.measure))]);
    }
  }, [queryData, selectedDimension, selectedValue, processedData]);

  // Initialize component with valid state on mount
  useEffect(() => {
    // Ensure we start with a valid combination
    if (!isValidDimensionValue(selectedDimension, selectedValue)) {
      const allDimensions = hesitancyInsightsApi.getAllDimensions();
      const availableValues = allDimensions[selectedDimension] || [];
      if (availableValues.length > 0) {
        setSelectedValue(availableValues[0]);
      }
    }
  }, []); // Only run once on mount

  const renderVisualization = () => {
    if (isLoading) {
      return <Loading />;
    }

    if (!processedData || processedData.length === 0) {
      return (
        <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg">
          <div className="text-center text-gray-500">
            <div className="text-4xl mb-2">📊</div>
            <p>No data available for the selected criteria</p>
            {!isValidDimensionValue(selectedDimension, selectedValue) && (
              <p className="text-sm mt-2 text-red-600">
                Invalid combination: {selectedDimension} = {selectedValue}
              </p>
            )}
          </div>
        </div>
      );
    }

    switch (viewType) {
      case 'map':
        if (selectedDimension === 'region') {
          return (
            <UKMap 
              data={processedData}
              title={`Vaccine Hesitancy by Region`}
              selectedRegion={selectedValue}
              onRegionClick={(region) => setSelectedValue(region)}
            />
          );
        }
        // Fallback to chart if not region dimension
        return (
          <VHBarChart 
            data={processedData}
            title={`Vaccine Measures for ${hesitancyInsightsApi.getDimensionDisplayNames()[selectedDimension] || selectedDimension}: ${selectedValue}`}
          />
        );

      case 'chart':
        return (
          <VHBarChart 
            data={processedData}
            title={`Vaccine Measures for ${hesitancyInsightsApi.getDimensionDisplayNames()[selectedDimension] || selectedDimension}: ${selectedValue}`}
          />
        );

      case 'table':
        return (
          <DataTable 
            data={queryData || []}
            title={`Vaccine Hesitancy Data - ${hesitancyInsightsApi.getDimensionDisplayNames()[selectedDimension] || selectedDimension}: ${selectedValue}`}
          />
        );

      default:
        return null;
    }
  };

  const getViewTypeOptions = () => {
    if (selectedDimension === 'region') {
      return [
        { value: 'map', label: 'Map View' },
        { value: 'chart', label: 'Chart View' },
        { value: 'table', label: 'Data Table' }
      ];
    } else {
      return [
        { value: 'chart', label: 'Chart View' },
        { value: 'table', label: 'Data Table' }
      ];
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          UK Vaccine Hesitancy Health Board
        </h1>
        <p className="text-gray-600">
          Interactive dashboard showing vaccine hesitancy data across different demographics and regions
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Dimension Selector */}
          <div>
            <Select
              label="Demographic Dimension"
              value={selectedDimension}
              onChange={(e) => handleDimensionChange(e.target.value)}
              options={Object.entries(hesitancyInsightsApi.getDimensionDisplayNames()).map(([key, display]) => ({
                value: key,
                label: display
              }))}
            />
          </div>

          {/* Value Selector */}
          <div>
            <Select
              label={`${hesitancyInsightsApi.getDimensionDisplayNames()[selectedDimension]} Value`}
              value={selectedValue}
              onChange={(e) => setSelectedValue(e.target.value)}
              options={getAvailableValues().map((value) => ({
                value: value,
                label: value
              }))}
            />
          </div>

          {/* Subgroup Selector for age_sex */}
          {selectedDimension === 'age_sex' && (
            <div>
              <Select
                label="Sex"
                value={selectedSubgroup}
                onChange={(e) => setSelectedSubgroup(e.target.value)}
                options={[
                  { value: '',      label: 'Both'   },
                  { value: 'Men',   label: 'Men'    },
                  { value: 'Women', label: 'Women'  },
                ]}
              />
            </div>
          )}

          {/* Group Selector for barriers */}
          {selectedDimension === 'barriers' && (
            <div>
              <Select
                label="Age Band"
                value={selectedBarrierGroup}
                onChange={(e) => setSelectedBarrierGroup(e.target.value)}
                options={[
                  { value: '', label: 'All bands' },
                  ...hesitancyInsightsApi.getAvailableBarrierGroups().map(g => ({
                    value: g,
                    label: g
                  }))
                ]}
              />
            </div>
          )}

          {/* View Type Selector */}
          <div>
            <Select
              label="View Type"
              value={viewType}
              onChange={(e) => setViewType(e.target.value as ViewType)}
              options={getViewTypeOptions()}
            />
          </div>

          {/* Quick Actions */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Quick Actions
            </label>
            <div className="flex space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleQuickRegionalMap}
              >
                Regional Map
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Visualization */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        {renderVisualization()}
      </div>

      {/* Data Summary */}
      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 text-gray-900">Data Summary</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-blue-50 p-4 rounded-lg">
            <h4 className="font-medium text-blue-900">Raw Records</h4>
            <p className="text-2xl font-bold text-blue-700">
              {queryData?.length || 0}
            </p>
            <p className="text-xs text-blue-600">Total from API</p>
          </div>
          <div className="bg-indigo-50 p-4 rounded-lg">
            <h4 className="font-medium text-indigo-900">Filtered Records</h4>
            <p className="text-2xl font-bold text-indigo-700">
              {processedData?.length || 0}
            </p>
            <p className="text-xs text-indigo-600">For visualization</p>
          </div>
          <div className="bg-green-50 p-4 rounded-lg">
            <h4 className="font-medium text-green-900">Current Selection</h4>
            <p className="text-sm text-green-700">
              {hesitancyInsightsApi.getDimensionDisplayNames()[selectedDimension]}: {selectedValue}
            </p>
            <p className="text-xs mt-1 text-green-600">
              {isValidDimensionValue(selectedDimension, selectedValue) ? 'Valid combination' : 'Invalid combination'}
            </p>
          </div>
          <div className="bg-purple-50 p-4 rounded-lg">
            <h4 className="font-medium text-purple-900">View Mode</h4>
            <p className="text-sm text-purple-700 capitalize">
              {viewType} View
            </p>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {queryError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-red-900 mb-2">Error Loading Data</h3>
          <p className="text-red-700">
            {queryError instanceof Error ? queryError.message : 'An unexpected error occurred'}
          </p>
          <p className="text-sm text-red-600 mt-2">
            Current selection: {selectedDimension} = {selectedValue}
          </p>
        </div>
      )}
    </div>
  );
} 