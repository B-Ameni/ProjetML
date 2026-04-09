import React, { useState, useEffect } from 'react';
import { Upload, Trash2 } from 'lucide-react';
import { getDataset, uploadDataset, cleanDataset } from '../services/api';

const Dashboard = () => {
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [previewData, setPreviewData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [columns, setColumns] = useState([]);
  const [selectedColumns, setSelectedColumns] = useState([]);
  const [outcomeFilter, setOutcomeFilter] = useState('all');
  const [uploading, setUploading] = useState(false);
  const [feedback, setFeedback] = useState('');

  const loadDataset = async () => {
    setLoading(true);
    try {
      const response = await getDataset();
      setDatasetInfo(response.data.info);
      setPreviewData(response.data.preview);
      setColumns(response.data.info.available_columns || []);
      setSelectedColumns(response.data.info.available_columns ? response.data.info.available_columns.slice(0, 5) : []);
    } catch (error) {
      console.error('Error fetching dataset:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDataset();
  }, []);

  const handleClean = async () => {
    try {
      setLoading(true);
      const response = await cleanDataset();
      setFeedback(response.data?.message || 'Dataset nettoyé.');
      await loadDataset();
    } catch (error) {
      console.error('Clean error:', error);
      setFeedback('Erreur lors du nettoyage.');
    } finally {
      setLoading(false);
      setTimeout(() => setFeedback(''), 4000);
    }
  };

  const handleUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      setUploading(true);
      const response = await uploadDataset(file);
      setFeedback(response.data?.message || 'Dataset téléversé.');
      await loadDataset();
    } catch (error) {
      console.error('Upload error:', error);
      setFeedback('Erreur lors du téléversement.');
    } finally {
      setUploading(false);
      setTimeout(() => setFeedback(''), 4000);
    }
  };

  const filteredPreview = previewData.filter((row) => {
    if (outcomeFilter === 'hit') return row.is_success === 1 || row.is_success === '1' || row.is_success === true;
    if (outcomeFilter === 'flop') return row.is_success === 0 || row.is_success === '0' || row.is_success === false;
    return true;
  }).map((row) => {
    if (selectedColumns.length === 0) return row;
    const filteredRow = {};
    selectedColumns.forEach(col => {
      filteredRow[col] = row[col];
    });
    return filteredRow;
  });

  return (
    <div className="space-y-8">
      <header className="flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Dataset</h1>
        </div>
        <div className="flex flex-wrap gap-3 items-center">
          <label className="flex items-center gap-2 px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg cursor-pointer text-sm font-bold transition-all">
            <Upload className="w-4 h-4" />
            <span>{uploading ? 'Téléversement...' : 'Importer'}</span>
            <input type="file" accept=".csv" onChange={handleUpload} className="hidden" />
          </label>
          <button
            onClick={handleClean}
            disabled={loading}
            className="flex items-center gap-2 px-5 py-2 bg-slate-600 hover:bg-slate-700 text-white font-bold rounded-lg transition-all disabled:opacity-60"
          >
            <Trash2 className="w-4 h-4" />
            Nettoyer
          </button>
          {feedback && <span className="text-sm text-slate-600">{feedback}</span>}
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Dataset Info Section */}
        <div className="lg:col-span-4 space-y-4">
          <h3 className="text-sm font-bold text-slate-600 uppercase tracking-widest">
            Informations
          </h3>

          {loading ? (
            <div className="p-6 bg-white border border-slate-200 rounded-lg shadow-sm">
              <p className="text-slate-500">Chargement...</p>
            </div>
          ) : datasetInfo ? (
            <div className="p-6 bg-white border border-slate-200 rounded-lg shadow-sm">
              <div className="mb-6">
                <p className="text-sm font-bold text-slate-900">{datasetInfo.name}</p>
                <p className="text-xs text-slate-500">{datasetInfo.size} • {datasetInfo.lines} films</p>
              </div>

              <div className="space-y-3">
                <div className="flex flex-col gap-3 text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-600">Colonnes</span>
                    <span className="font-bold text-slate-700">{datasetInfo.columns}</span>
                  </div>
                  {datasetInfo.counts && (
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="rounded-lg bg-slate-50 p-3 border border-slate-100">
                        <p className="font-bold text-slate-800">Succès</p>
                        <p>{datasetInfo.counts.hit ?? '—'}</p>
                      </div>
                      <div className="rounded-lg bg-slate-50 p-3 border border-slate-100">
                        <p className="font-bold text-slate-800">Flops</p>
                        <p>{datasetInfo.counts.flop ?? '—'}</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="p-6 bg-white border border-slate-200 rounded-lg shadow-sm">
              <p className="text-slate-500">Erreur de chargement.</p>
            </div>
          )}

        </div>

        {/* Data Preview & Actions */}
        <div className="lg:col-span-8 space-y-4">
          <div className="bg-white p-4 border border-slate-200 rounded-lg shadow-sm space-y-4">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <h3 className="font-bold text-slate-900">Prévisualisation</h3>
                <p className="text-sm text-slate-500">Filtrez les résultats et choisissez les colonnes affichées.</p>
              </div>
              <button
                onClick={handleClean}
                disabled={loading}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all disabled:opacity-60"
              >
                Nettoyer
              </button>
            </div>

            <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
              <div className="space-y-2">
                <label className="text-xs font-bold text-slate-600">Filtrer par résultat</label>
                <select
                  value={outcomeFilter}
                  onChange={(e) => setOutcomeFilter(e.target.value)}
                  className="w-full px-4 py-2 border border-slate-200 rounded-lg bg-white text-sm text-slate-700"
                >
                  <option value="all">Tous</option>
                  <option value="hit">Succès</option>
                  <option value="flop">Flop</option>
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-xs font-bold text-slate-600">Colonnes</label>
                <div className="grid grid-cols-2 gap-2">
                  {columns.slice(0, 8).map((col) => (
                    <button
                      key={col}
                      type="button"
                      onClick={() => {
                        setSelectedColumns((prev) => prev.includes(col)
                          ? prev.filter((item) => item !== col)
                          : [...prev, col]);
                      }}
                      className={`text-xs px-3 py-2 rounded-lg border ${selectedColumns.includes(col) ? 'bg-blue-600 text-white border-blue-600' : 'bg-slate-50 text-slate-700 border-slate-200'}`}
                    >
                      {col}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-lg overflow-hidden shadow-sm">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-sm">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-200">
                    {(selectedColumns.length > 0 ? selectedColumns : ['title', 'budget', 'popularity', 'runtime', 'revenue']).map((col) => (
                      <th key={col} className="p-3 text-xs font-bold text-slate-600 uppercase">{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {filteredPreview.slice(0, 15).map((row, index) => (
                    <tr key={index} className="hover:bg-slate-50">
                      {(selectedColumns.length > 0 ? selectedColumns : ['title', 'budget', 'popularity', 'runtime', 'revenue']).map((col) => (
                        <td key={`${index}-${col}`} className="p-3 text-slate-900">
                          {row[col] === undefined ? '-' : typeof row[col] === 'number' ? row[col].toLocaleString() : row[col]}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;