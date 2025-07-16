import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import { config } from '@/lib/config';
import { ApiResponse } from '@/types';

// Default API timeout (in milliseconds)
const DEFAULT_TIMEOUT = 120000; // 2 minutes

export class BaseApiClient {
  protected client: AxiosInstance;

  constructor(baseURL: string) {
    this.client = axios.create({
      baseURL,
      // No global timeout here; set per-request
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('API Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        console.log(`API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error('API Response Error:', error);
        if (error.response) {
          // Server responded with error status
          console.error('Error Status:', error.response.status);
          console.error('Error Data:', error.response.data);
        } else if (error.request) {
          // Request was made but no response received
          console.error('No response received:', error.request);
        } else {
          // Something else happened
          console.error('Error:', error.message);
        }
        return Promise.reject(error);
      }
    );
  }

  protected async request<T>(config: AxiosRequestConfig & { timeout?: number } = {}): Promise<ApiResponse<T>> {
    try {
      // Use per-request timeout if provided, else default
      const timeout = config.timeout ?? DEFAULT_TIMEOUT;
      const response = await this.client.request<T>({ ...config, timeout });
      return {
        success: true,
        data: response.data,
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.message || 'An error occurred',
        data: error.response?.data,
      };
    }
  }

  protected async get<T>(url: string, config?: AxiosRequestConfig & { timeout?: number }): Promise<ApiResponse<T>> {
    return this.request<T>({ ...config, method: 'GET', url });
  }

  protected async post<T>(url: string, data?: any, config?: AxiosRequestConfig & { timeout?: number }): Promise<ApiResponse<T>> {
    return this.request<T>({ ...config, method: 'POST', url, data });
  }

  protected async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>({ ...config, method: 'PUT', url, data });
  }

  protected async delete<T>(url: string, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>({ ...config, method: 'DELETE', url });
  }

  // Health check method for all agents
  async healthCheck(): Promise<ApiResponse<{ status: string }>> {
    return this.get<{ status: string }>('/health');
  }
} 