import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:8000"
});

export const trainModel = (data) => API.post("/train", data);
export const tuneModel = (data) => API.post("/tune", data);
export const getResults = () => API.get("/results");
export const getModels = () => API.get("/models");
export const getDataset = () => API.get("/dataset");
export const uploadDataset = (file) => {
  const formData = new FormData();
  formData.append('file', file);
  return API.post('/dataset/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
};
export const cleanDataset = () => API.post('/dataset/clean');
export const predict = (data) => API.post("/predict", data);
export const rollback = (runId) => API.post(`/rollback/${runId}`);
export const exportModel = (runId) => API.get(`/export-model/${runId}`, { responseType: 'blob' });