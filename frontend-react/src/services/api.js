import axios from 'axios';

let API_BASE_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:8000';

export const setApiBaseUrl = (url) => {
  API_BASE_URL = url;
};

export const getApiBaseUrl = () => API_BASE_URL;

const getApi = () => axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Health check
export const healthCheck = async () => {
  try {
    const response = await getApi().get('/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
};

export const getModelPerformance = async () => {
  try {
    const response = await getApi().get('/model_performance');
    return response.data;
  } catch (error) {
    console.error('Error fetching model performance:', error);
    throw error;
  }
};

export const analyzeImage = async (file) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axios.post(`${API_BASE_URL}/analyze_image`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error analyzing image:', error);
    throw error;
  }
};

export const analyzeThermal = async (file) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axios.post(`${API_BASE_URL}/analyze_thermal`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error analyzing thermal image:', error);
    throw error;
  }
};

export const predictRisk = async (data) => {
  try {
    const response = await getApi().post('/predict_risk', data);
    return response.data;
  } catch (error) {
    console.error('Error predicting risk:', error);
    throw error;
  }
};

export const explainRisk = async (data) => {
  try {
    const response = await getApi().post('/explain_risk', data);
    return response.data;
  } catch (error) {
    console.error('Error explaining risk:', error);
    throw error;
  }
};

export const analyzeStructure = async (data) => {
  try {
    const response = await getApi().post('/analyze_structure', data);
    return response.data;
  } catch (error) {
    console.error('Error analyzing structure:', error);
    throw error;
  }
};

export const generateReportUrl = () => {
  return `${API_BASE_URL}/generate_report`;
};

export const chatWithInspector = async (data) => {
  try {
    const response = await getApi().post('/chat', data);
    return response.data;
  } catch (error) {
    console.error('Error chatting with inspector:', error);
    throw error;
  }
};
