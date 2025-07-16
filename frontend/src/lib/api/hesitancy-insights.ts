import { BaseApiClient } from './base';
import { config } from '@/lib/config';
import { ApiResponse } from '@/types';

// New API types for the generic endpoint
export interface VHGenericRequest {
  dimension: string;
  value: string;
  filters?: Record<string, string>;
}

export interface VHGenericResponse {
  data: Array<Record<string, any>>;
  sql: string;
}

export interface VHDimensionsResponse {
  dimensions: string[];
  tables: Record<string, string>;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  mcp_connected: boolean;
}

export interface VHDataPoint {
  region: string;
  measure: string;
  percent: number;
}

export interface VHResponse {
  data: {
    data: VHDataPoint[];
  };
}

export class HesitancyInsightsApi extends BaseApiClient {
  constructor() {
    super(config.agents.hesitancyInsights);
  }

  // Generic vaccine hesitancy data query
  async getVHData(request: VHGenericRequest): Promise<ApiResponse<VHGenericResponse>> {
    return this.post<VHGenericResponse>('/vh', request);
  }

  // Get available dimensions
  async getDimensions(): Promise<ApiResponse<VHDimensionsResponse>> {
    return this.get<VHDimensionsResponse>('/dimensions');
  }

  // Health check
  async getHealth(): Promise<ApiResponse<HealthResponse>> {
    return this.get<HealthResponse>('/health');
  }

  // Convenience methods for specific dimensions
  async getRegionData(region: string, filters?: Record<string, string>): Promise<ApiResponse<VHGenericResponse>> {
    return this.getVHData({ dimension: 'region', value: region, filters });
  }

  async getSexData(sex: string, filters?: Record<string, string>): Promise<ApiResponse<VHGenericResponse>> {
    return this.getVHData({ dimension: 'sex', value: sex, filters });
  }

  async getEthnicityData(ethnicity: string, filters?: Record<string, string>): Promise<ApiResponse<VHGenericResponse>> {
    return this.getVHData({ dimension: 'ethnicity', value: ethnicity, filters });
  }

  async getAgeBandData(ageBand: string, filters?: Record<string, string>): Promise<ApiResponse<VHGenericResponse>> {
    return this.getVHData({ dimension: 'age_band', value: ageBand, filters });
  }

  async getAgeGroupData(ageGroup: string, filters?: Record<string, string>): Promise<ApiResponse<VHGenericResponse>> {
    return this.getVHData({ dimension: 'age_group', value: ageGroup, filters });
  }

  async getReligionData(religion: string, filters?: Record<string, string>): Promise<ApiResponse<VHGenericResponse>> {
    return this.getVHData({ dimension: 'religion', value: religion, filters });
  }

  async getDisabilityData(status: string, filters?: Record<string, string>): Promise<ApiResponse<VHGenericResponse>> {
    return this.getVHData({ dimension: 'disability_status', value: status, filters });
  }

  async getCEVData(status: string, filters?: Record<string, string>): Promise<ApiResponse<VHGenericResponse>> {
    return this.getVHData({ dimension: 'cev_status', value: status, filters });
  }

  async getHealthConditionData(condition: string, filters?: Record<string, string>): Promise<ApiResponse<VHGenericResponse>> {
    return this.getVHData({ dimension: 'health_condition', value: condition, filters });
  }

  async getHealthGeneralData(condition: string, filters?: Record<string, string>): Promise<ApiResponse<VHGenericResponse>> {
    return this.getVHData({ dimension: 'health_general_condition', value: condition, filters });
  }

  async getIMDData(quintile: string, filters?: Record<string, string>): Promise<ApiResponse<VHGenericResponse>> {
    return this.getVHData({ dimension: 'imd_quintile', value: quintile, filters });
  }

  async getEmploymentData(status: string, filters?: Record<string, string>): Promise<ApiResponse<VHGenericResponse>> {
    return this.getVHData({ dimension: 'employment_status', value: status, filters });
  }

  async getExpenseData(affordability: string, filters?: Record<string, string>): Promise<ApiResponse<VHGenericResponse>> {
    return this.getVHData({ dimension: 'expense_affordability', value: affordability, filters });
  }

  async getHouseholdData(type: string, filters?: Record<string, string>): Promise<ApiResponse<VHGenericResponse>> {
    return this.getVHData({ dimension: 'household_type', value: type, filters });
  }

  async getCaregiverData(status: string, filters?: Record<string, string>): Promise<ApiResponse<VHGenericResponse>> {
    return this.getVHData({ dimension: 'caregiver_status', value: status, filters });
  }

  async getAgeSexData(ageBand: string, subgroup?: "Men" | "Women"): Promise<ApiResponse<VHGenericResponse>> {
    const filters = subgroup ? { subgroup } : undefined;
    return this.getVHData({ dimension: "age_sex", value: ageBand, filters });
  }

  async getTrendsData(period: string, filters?: Record<string, string>): Promise<ApiResponse<VHGenericResponse>> {
    return this.getVHData({ dimension: 'trends', value: period, filters });
  }

  async getBarriersData(block: string, group?: string): Promise<ApiResponse<VHGenericResponse>> {
    const filters = group ? { group } : undefined;
    return this.getVHData({ dimension: "barriers", value: block, filters });
  }

  async getReasonsData(block: string, filters?: Record<string, string>): Promise<ApiResponse<VHGenericResponse>> {
    return this.getVHData({ dimension: 'reasons', value: block, filters });
  }

  // Static data for form options (from the backend validation)
  getAvailableRegions(): string[] {
    return [
      "All adults",
      "East Midlands",
      "East of England", 
      "England",
      "London",
      "North East",
      "North West",
      "Scotland",
      "South East",
      "South West",
      "Wales",
      "West Midlands",
      "Yorkshire and The Humber",
    ];
  }

  getAvailableSex(): string[] {
    return ["All adults", "Men", "Women"];
  }

  getAvailableEthnicity(): string[] {
    return [
      "All adults",
      "Black or Black British",
      "Mixed or multiple ethnic groups",
      "Other ethnic group",
      "White",
      "White - British",
      "White - Other White background",
    ];
  }

  getAvailableAgeBands(): string[] {
    return [
      "Aged 16 to 29",
      "Aged 30 to 49", 
      "Aged 50 and over",
      "All adults",
    ];
  }

  getAvailableAgeGroups(): string[] {
    return [
      "Aged 16 to 171",
      "Aged 18 to 21",
      "Aged 22 to 25",
      "Aged 26 to 29",
      "Aged 30 to 39",
      "Aged 40 to 49",
      "Aged 50 to 59",
      "Aged 60 to 69",
      "Aged 70 and above",
      "All adults",
    ];
  }

  getAvailableReligions(): string[] {
    return [
      "All adults",
      "Buddhist",
      "Christian",
      "Hindu",
      "Jewish",
      "Muslim",
      "No religion",
      "Other religion",
      "Prefer not to say",
      "Sikh",
    ];
  }

  getAvailableAgeSexBands(): string[] {
    return [
      "16 to 29",
      "30 to 49",
      "Aged 50 and over",
      "All adults",
    ];
  }

  getAvailableAgeSexSubgroups(): string[] {
    return ["Men", "Women"];
  }

  getAvailableTrendPeriods(): string[] {
    return [
      "10 December to 10 January5",
      "13 January to 7 February",
      "17 February to 14 March",
      "31 March to 25 April",
      "28 April to 23 May",
      "26 May to 20 June",
      "23 June to 18 July",
    ];
  }

  getAvailableBarrierBlocks(): string[] {
    return [
      "Among those who have received at least one dose of a vaccine",
      "Among those who have not yet received a vaccine",
    ];
  }

  getAvailableBarrierGroups(): string[] {
    return [
      "All persons",
      "Aged 16 to 29",
      "Aged 30 to 49",
      "Aged 50 and over",
      "Men",
      "Women",
      "Disabled",
      "Non-disabled",
      "Don't know/Prefer not to say",
    ];
  }

  getAvailableReasons(): string[] {
    return [
      "Health2",
      "Catching the coronavirus (COVID-19)2",
      "Fertility2",
      "General hesitation about the vaccine and its safety2",
      "Not needed (now or ever)2",
      "Travel and 'other' reasons2"
    ];
  }

  // Get all available dimensions and their values
  getAllDimensions(): Record<string, string[]> {
    return {
      region: this.getAvailableRegions(),
      sex: this.getAvailableSex(),
      ethnicity: this.getAvailableEthnicity(),
      age_band: this.getAvailableAgeBands(),
      age_group: this.getAvailableAgeGroups(),
      religion: this.getAvailableReligions(),
      disability_status: ["All adults", "Disabled", "Not disabled", "Don't know/Prefer not to say"],
      cev_status: ["All adults", "CEV", "Not CEV"],
      health_condition: ["All adults", "Health condition", "No health condition", "Don't know", "Prefer not to say"],
      health_general_condition: ["All adults", "Good/Very good health", "Fair health", "Bad/Very bad health", "Don't know", "Prefer not to say"],
      imd_quintile: ["All adults in England", "Most deprived", "2nd quintile", "3rd quintile", "4th quintile", "Least deprived"],
      employment_status: ["All adults", "Employed / self-employed", "In employment1", "Unemployed", "Unpaid family worker", "Economically inactive - retired", "Economically inactive - other"],
      expense_affordability: ["Able to afford an unexpected, but necessary, expense of £850", "All adults", "Don't know/Prefer not to say", "Unable to afford an unexpected, but necessary, expense of £850"],
      household_type: ["All adults", "One adult living alone", "Three or more people", "Two person household"],
      caregiver_status: ["All adults", "Cares for someone in their own home1", "Doesn't care for someone in their own home1", "Don't know/Prefer not to say"],
      age_sex: this.getAvailableAgeSexBands(),
      trends: this.getAvailableTrendPeriods(),
      barriers: this.getAvailableBarrierBlocks(),
      barriers_group: this.getAvailableBarrierGroups(),
      reasons: this.getAvailableReasons(),
    };
  }

  // Get dimension display names
  getDimensionDisplayNames(): Record<string, string> {
    return {
      region: "Region",
      sex: "Sex",
      ethnicity: "Ethnicity", 
      age_band: "Age Band",
      age_group: "Age Group",
      religion: "Religion",
      disability_status: "Disability Status",
      cev_status: "CEV Status",
      health_condition: "Health Condition",
      health_general_condition: "General Health",
      imd_quintile: "IMD Quintile",
      employment_status: "Employment Status",
      expense_affordability: "Expense Affordability",
      household_type: "Household Type",
      caregiver_status: "Caregiver Status",
      age_sex: "Age & Sex",
      trends: "Trends",
      barriers: "Barriers",
      reasons: "Reasons",
    };
  }
}

// Export singleton instance
export const hesitancyInsightsApi = new HesitancyInsightsApi();

// Add new function for fetching all regions data for map
export async function fetchAllRegionsData(): Promise<VHDataPoint[]> {
  const response = await hesitancyInsightsApi.getVHData({
    dimension: 'region',
    value: 'all_regions'
  })
  
  if (!response.data?.data) {
    return []
  }
  
  // Filter to only include "Vaccine hesitancy1,2" measure for map visualization
  return response.data.data
    .filter((item: any) => 
      item.measure === 'Vaccine hesitancy1,2' && 
      item.percent !== null
    )
    .map((item: any) => ({
      region: item.region,
      measure: item.measure,
      percent: item.percent
    }))
}

// Helper function to normalize region names between API and GeoJSON
export function normalizeRegionName(name: string): string {
  // Remove parentheticals like "(England)" from GeoJSON names
  return name.replace(/\s*\([^)]*\)$/, '').trim()
}

// Helper function to build region-to-value mapping for choropleth
export function buildRegionValueMap(data: VHDataPoint[], measure: string = 'Vaccine hesitancy1,2'): Record<string, number> {
  return Object.fromEntries(
    data
      .filter((item: VHDataPoint) => item.measure === measure && item.percent !== null)
      .map((item: VHDataPoint) => [item.region, item.percent])
  )
} 