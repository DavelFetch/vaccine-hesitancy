import React, { useEffect, useRef, useState } from 'react';
import mapboxgl, { GeoJSONSource } from 'mapbox-gl';
import type { Feature, FeatureCollection, MultiPolygon } from 'geojson';
import { feature as topojsonFeature } from 'topojson-client';
import 'mapbox-gl/dist/mapbox-gl.css';

interface RegionData {
  region: string;
  value: number; // Keep for backward compatibility (will be hesitancy percentage)
  hesitancyPercent?: number; // Vaccine hesitancy percentage
  sentimentPercent?: number; // Positive vaccine sentiment percentage
  measure?: string;
  [key: string]: any;
}

interface UKMapProps {
  data: RegionData[];
  title?: string;
  selectedRegion?: string;
  onRegionClick?: (region: string) => void;
}

interface RegionProperties {
  NUTS112NM: string;
  percent?: number;
  hesitancyPercent?: number;
  sentimentPercent?: number;
  [key: string]: any;
}

type UKRegionFeature = Feature<MultiPolygon, RegionProperties>;
type UKRegionCollection = FeatureCollection<MultiPolygon, RegionProperties>;

// Warn if Mapbox token is missing
if (!process.env.NEXT_PUBLIC_MAPBOX_TOKEN) {
  // eslint-disable-next-line no-console
  console.warn('⚠️ Mapbox token is missing! Set NEXT_PUBLIC_MAPBOX_TOKEN in your .env file.');
}

// Set your Mapbox access token
mapboxgl.accessToken = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || '';

// Function to map backend region names to GeoJSON region names
const mapRegionNameToGeoJSON = (backendRegionName: string): string | null => {
  const mapping: Record<string, string | null> = {
    "East Midlands": "East Midlands (England)",
    "North East": "North East (England)", 
    "North West": "North West (England)",
    "South East": "South East (England)",
    "South West": "South West (England)",
    "West Midlands": "West Midlands (England)",
    "East of England": "East of England",
    "London": "London",
    "Wales": "Wales",
    "Yorkshire and The Humber": "Yorkshire and The Humber",
    // Aggregates that don't exist in GeoJSON
    "All adults": null,
    "England": null,
    "Scotland": null, // Not in this GeoJSON
  };
  
  return mapping[backendRegionName] || null;
};

export function UKMap({ data, title, selectedRegion, onRegionClick }: UKMapProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const popup = useRef<mapboxgl.Popup | null>(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [showLoadTimeout, setShowLoadTimeout] = useState(false);

  // Visual loading spinner state
  const [showSpinner, setShowSpinner] = useState(true);

  // Debug: log when map container is mounted
  useEffect(() => {
    if (mapContainer.current) {
      // eslint-disable-next-line no-console
      console.log('🧩 Map container mounted:', mapContainer.current);
    }
  }, []);

  useEffect(() => {
    console.log('🟢 Map useEffect mount');
    if (!mapContainer.current || map.current) return;

    // Debug: log map initialization start
    // eslint-disable-next-line no-console
    console.log('🚦 Initializing Mapbox map...');

    // Initialize map
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/streets-v11',
      center: [-2.5, 54], // Center on UK
      zoom: 4.5,
      interactive: true
    });

    // Create popup but don't add to map yet
    popup.current = new mapboxgl.Popup({
      closeButton: false,
      closeOnClick: false
    });

    // Map load timeout warning
    const timeout = setTimeout(() => {
      if (!mapLoaded) {
        setShowLoadTimeout(true);
        // eslint-disable-next-line no-console
        console.warn('⏳ Map is taking longer than expected to load...');
      }
    }, 5000);

    // Load TopoJSON, convert to GeoJSON, and add as source
    map.current.on('load', async () => {
      try {
        let geojsonData = null;
        // Try loading TopoJSON first
        try {
          console.log('🌍 Fetching topo_nuts1.json...');
          const topoRes = await fetch('/topo_nuts1.json');
          if (!topoRes.ok) throw new Error('Failed to fetch topo_nuts1.json');
          const topoData = await topoRes.json();
          // Convert TopoJSON to GeoJSON
          // Assume the object key is the first key in topoData.objects
          const objectKey = Object.keys(topoData.objects)[0];
          geojsonData = topojsonFeature(topoData, topoData.objects[objectKey]);
          console.log('✅ TopoJSON loaded and converted to GeoJSON.');
        } catch (err) {
          console.warn('⚠️ Failed to load topo_nuts1.json, falling back to nuts1.json:', err);
          // Fallback to nuts1.json
          const geoRes = await fetch('/nuts1.json');
          if (!geoRes.ok) throw new Error('Failed to fetch nuts1.json');
          geojsonData = await geoRes.json();
        }

        // Add UK regions source
        map.current?.addSource('uk-nuts1', {
          type: 'geojson',
          data: geojsonData as any
        });

        // Add choropleth layer
        map.current?.addLayer({
          id: 'region-choropleth',
          type: 'fill',
          source: 'uk-nuts1',
          paint: {
            'fill-color': [
              'interpolate',
              ['linear'],
              ['get', 'percent'],
              0, '#f2f0f7',
              5, '#cbc9e2',
              10, '#9e9ac8',
              20, '#756bb1',
              30, '#54278f'
            ],
            'fill-opacity': 0.8
          }
        });

        // Add outline layer
        map.current?.addLayer({
          id: 'region-outline',
          type: 'line',
          source: 'uk-nuts1',
          paint: {
            'line-color': '#444',
            'line-width': 1
          }
        });

        // Add hover effects
        map.current?.on('mousemove', 'region-choropleth', (e) => {
          if (e.features?.[0]?.properties && map.current) {
            map.current.getCanvas().style.cursor = 'pointer';
            const props = e.features[0].properties;
            
            // Build tooltip content with both measures
            let tooltipContent = `
              <div class="bg-white p-3 rounded-lg shadow-lg border border-gray-200 min-w-48">
                <div class="font-semibold text-gray-900 mb-2 text-center border-b border-gray-200 pb-2">
                  ${props.NUTS112NM}
                </div>
            `;
            
            // Add vaccine hesitancy if available
            if (props.hesitancyPercent !== null && props.hesitancyPercent !== undefined) {
              tooltipContent += `
                <div class="flex justify-between items-center mb-1">
                  <span class="text-sm text-gray-700">Vaccine Hesitancy:</span>
                  <span class="font-medium text-red-600">${props.hesitancyPercent.toFixed(1)}%</span>
                </div>
              `;
            }
            
            // Add positive sentiment if available
            if (props.sentimentPercent !== null && props.sentimentPercent !== undefined) {
              tooltipContent += `
                <div class="flex justify-between items-center">
                  <span class="text-sm text-gray-700">Positive Sentiment:</span>
                  <span class="font-medium text-green-600">${props.sentimentPercent.toFixed(1)}%</span>
                </div>
              `;
            }
            
            // Fallback to basic percent if enhanced data not available
            if ((props.hesitancyPercent === null || props.hesitancyPercent === undefined) && 
                (props.sentimentPercent === null || props.sentimentPercent === undefined) && 
                props.percent !== null && props.percent !== undefined) {
              tooltipContent += `
                <div class="flex justify-between items-center">
                  <span class="text-sm text-gray-700">Value:</span>
                  <span class="font-medium text-blue-600">${props.percent.toFixed(1)}%</span>
                </div>
              `;
            }
            
            tooltipContent += `</div>`;
            
            popup.current
              ?.setLngLat(e.lngLat)
              .setHTML(tooltipContent)
              .addTo(map.current);
          }
        });

        map.current?.on('mouseleave', 'region-choropleth', () => {
          if (map.current) {
            map.current.getCanvas().style.cursor = '';
            popup.current?.remove();
          }
        });

        // Add click handler
        map.current?.on('click', 'region-choropleth', (e) => {
          if (e.features?.[0]?.properties && onRegionClick) {
            const region = e.features[0].properties.NUTS112NM;
            onRegionClick(region);
          }
        });

        setMapLoaded(true);
        setShowSpinner(false);
        setShowLoadTimeout(false);
        clearTimeout(timeout);
        console.log('🗺️ Map loaded!');
      } catch (error) {
        setShowSpinner(false);
        setShowLoadTimeout(false);
        clearTimeout(timeout);
        console.error('❌ Error loading map data:', error);
      }
    });

    // Cleanup
    return () => {
      console.log('🔴 Map useEffect cleanup');
      popup.current?.remove();
      map.current?.remove();
      clearTimeout(timeout);
    };
  }, []);

  // If the map is in a tab/panel, call resize when it becomes visible
  useEffect(() => {
    if (map.current && mapLoaded) {
      setTimeout(() => {
        try {
          map.current?.resize();
          // eslint-disable-next-line no-console
          console.log('🗺️ Mapbox map.resize() called');
        } catch (e) {
          // eslint-disable-next-line no-console
          console.warn('Mapbox resize error:', e);
        }
      }, 300);
    }
  }, [mapLoaded]);

  // Debug: log when data is being processed
  useEffect(() => {
    // eslint-disable-next-line no-console
    console.log('🔄 Processing map data:', data);
  }, [data]);

  // Update data when it changes
  useEffect(() => {
    if (!map.current || !mapLoaded || !data || data.length === 0) return;

    try {
      console.log('UKMap updating data:', data);
      
      const source = map.current.getSource('uk-nuts1') as GeoJSONSource;
      if (!source) return;

      // Get the current GeoJSON data
      const currentData = (source as any)._data as UKRegionCollection;
      if (!currentData?.features) return;

      console.log('Available GeoJSON regions:', currentData.features.map(f => f.properties.NUTS112NM));
      console.log('Available API regions:', data.map(d => d.region));

      const updatedFeatures = currentData.features.map((feature: UKRegionFeature) => {
        // Try exact match first
        let regionData = data.find(d => d.region === feature.properties.NUTS112NM);
        
        // If no exact match, try normalized matching
        if (!regionData) {
          const normalizedFeatureName = feature.properties.NUTS112NM.replace(/\s*\([^)]*\)$/, '').trim();
          regionData = data.find(d => d.region === normalizedFeatureName);
        }
        
        const percentValue = regionData?.value || regionData?.percent || 0;
        const hesitancyValue = regionData?.hesitancyPercent || null;
        const sentimentValue = regionData?.sentimentPercent || null;
        
        console.log(`Mapping ${feature.properties.NUTS112NM} -> Hesitancy: ${hesitancyValue}%, Sentiment: ${sentimentValue}%`);
        
        return {
          ...feature,
          properties: {
            ...feature.properties,
            percent: percentValue,
            hesitancyPercent: hesitancyValue,
            sentimentPercent: sentimentValue
          }
        } as UKRegionFeature;
      });

      // Update source data
      source.setData({
        type: 'FeatureCollection',
        features: updatedFeatures
      } as UKRegionCollection);
      
      console.log('Map data updated successfully');
    } catch (error) {
      console.error('Error updating map data:', error);
    }
  }, [data, mapLoaded]);

  // Update selected region
  useEffect(() => {
    if (!map.current || !mapLoaded) return;

    // Reset any previous filters
    map.current.setFilter('region-choropleth', null);
    map.current.setPaintProperty('region-choropleth', 'fill-opacity', 0.8);

    if (selectedRegion) {
      // Map backend region name to GeoJSON region name
      const mappedRegionName = mapRegionNameToGeoJSON(selectedRegion);
      
      if (mappedRegionName) {
        // Highlight selected region using mapped name
        map.current.setFilter('region-choropleth', [
          '==',
          ['get', 'NUTS112NM'],
          mappedRegionName
        ]);
        map.current.setPaintProperty('region-choropleth', 'fill-opacity', 1);
      }
      // If mappedRegionName is null (aggregate regions), don't highlight anything
    }
  }, [selectedRegion, mapLoaded]);

  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg">
        <div className="text-center text-gray-500">
          <div className="text-4xl mb-2">🗺️</div>
          <p>No regional data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full">
      {title && (
        <h3 className="text-lg font-semibold mb-4 text-gray-900">{title}</h3>
      )}
      
      <div className="relative bg-white border border-gray-200 rounded-lg p-4">
        {/* Loading Spinner Overlay */}
        {showSpinner && (
          <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-70 z-20">
            <svg className="animate-spin h-8 w-8 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path>
            </svg>
            <span className="ml-3 text-blue-700 font-medium">Loading map...</span>
          </div>
        )}
        {/* Timeout warning */}
        {showLoadTimeout && (
          <div className="absolute inset-0 flex items-center justify-center bg-yellow-100 bg-opacity-80 z-30">
            <span className="text-yellow-800 font-semibold">Map is taking longer than expected to load...</span>
          </div>
        )}
        <div ref={mapContainer} className="w-full h-80" />
        
        {/* Legend */}
        <div className="mt-4 flex items-center justify-between">
          <div className="text-sm text-gray-600">
            Vaccine Hesitancy Rate by Region
          </div>
          <div className="flex items-center space-x-2 text-sm text-gray-600">
            <span>0%</span>
            <div className="w-32 h-3 bg-gradient-to-r from-[#f2f0f7] via-[#9e9ac8] to-[#54278f] rounded"></div>
            <span>30%+</span>
          </div>
        </div>
      </div>
    </div>
  );
} 