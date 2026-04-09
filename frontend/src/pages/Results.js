import React, { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area
} from 'recharts';
import { Download, FileText } from 'lucide-react';

import { getResults } from '../services/api';

const ROC_DATA = [
  { fpr: 0, tpr: 0 },
  { fpr: 0.1, tpr: 0.45 },
  { fpr: 0.2, tpr: 0.75 },
  { fpr: 0.4, tpr: 0.88 },
  { fpr: 0.6, tpr: 0.94 },
  { fpr: 0.8, tpr: 0.98 },
  { fpr: 1, tpr: 1 },
];

const ConfusionMatrix = ({ cm }) => (
  <div className="flex flex-col items-center">
    <div className="grid grid-cols-3 gap-1">
      <div className="w-12 h-12 flex items-center justify-center text-[10px] font-bold text-slate-400 p-2 text-center border-b border-r border-slate-100">Actual</div>
      <div className="w-20 h-12 flex items-center justify-center text-xs font-bold text-slate-500">Hit</div>
      <div className="w-20 h-12 flex items-center justify-center text-xs font-bold text-slate-500">Flop</div>

      <div className="w-12 h-20 flex items-center justify-center text-xs font-bold text-slate-500 -rotate-90">Hit</div>
      <div className="w-20 h-20 bg-green-100 flex items-center justify-center text-lg font-bold text-green-700 border border-green-300">
        {cm?.tp || 0}
      </div>
      <div className="w-20 h-20 bg-red-100 flex items-center justify-center text-lg font-bold text-red-700 border border-red-300">
        {cm?.fn || 0}
      </div>

      <div className="w-12 h-20 flex items-center justify-center text-xs font-bold text-slate-500 -rotate-90">Flop</div>
      <div className="w-20 h-20 bg-red-100 flex items-center justify-center text-lg font-bold text-red-700 border border-red-300">
        {cm?.fp || 0}
      </div>
      <div className="w-20 h-20 bg-green-100 flex items-center justify-center text-lg font-bold text-green-700 border border-green-300">
        {cm?.tn || 0}
      </div>
    </div>
  </div>
);

const Results = () => {
  const [models, setModels] = useState([]);
  const [cmData, setCmData] = useState({ tp: 0, tn: 0, fp: 0, fn: 0 });

  useEffect(() => {
    getResults().then(res => {
      const data = res.data || {};
      setModels(data.models || []);
      setCmData(data.confusion_matrix || { tp: 0, tn: 0, fp: 0, fn: 0 });
    }).catch(err => console.error(err));
  }, []);

  const handleExportCSV = () => {
    if (models.length === 0) {
      alert('Aucun résultat à exporter.');
      return;
    }
    const csvRows = [
      ['Model', 'Status', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Dataset', 'Version'],
    ];
    models.forEach((model) => {
      csvRows.push([
        model.model,
        model.status,
        `${(model.metrics.accuracy * 100).toFixed(1)}%`,
        `${(model.metrics.precision * 100).toFixed(1)}%`,
        `${(model.metrics.recall * 100).toFixed(1)}%`,
        `${(model.metrics.f1_score * 100).toFixed(1)}%`,
        model.dataset,
        model.version,
      ]);
    });
    const csvContent = csvRows.map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'model_results.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-8">
      <header className="flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Résultats</h1>
        </div>
        <div className="flex flex-wrap gap-3">
          <button
            onClick={handleExportCSV}
            className="flex items-center gap-2 px-5 py-2.5 bg-indigo-50 border border-indigo-100 text-indigo-700 font-bold rounded-lg hover:bg-indigo-100 transition-all shadow-sm active:scale-95"
          >
            <FileText className="w-4 h-4 text-indigo-400" />
            Exporter CSV
          </button>
          <button
            onClick={() => alert('L’export PNG est disponible via la capture de l’écran de cette interface.')}
            className="flex items-center gap-2 px-5 py-2.5 bg-white border border-slate-200 text-slate-700 font-bold rounded-lg hover:bg-slate-50 transition-all shadow-sm active:scale-95"
          >
            <Download className="w-4 h-4 text-slate-400" />
            Exporter PNG
          </button>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Confusion Matrix Card */}
        <div className="p-8 bg-white border border-slate-200 rounded-lg shadow-sm space-y-6">
          <h3 className="font-bold text-slate-900 border-b border-slate-100 pb-4">
            Matrice de Confusion
          </h3>
          <div className="py-8 flex justify-center">
            <ConfusionMatrix cm={cmData} />
          </div>
        </div>

        {/* ROC Curve Card */}
        <div className="p-8 bg-white border border-slate-200 rounded-lg shadow-sm space-y-6">
          <h3 className="font-bold text-slate-900 border-b border-slate-100 pb-4">
            Courbes ROC (AUC: 0.94)
          </h3>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={ROC_DATA}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis dataKey="fpr" stroke="#94a3b8" fontSize={10} label={{ value: 'False Positive Rate (FPR)', position: 'bottom', offset: 0, fontSize: 10 }} />
                <YAxis stroke="#94a3b8" fontSize={10} label={{ value: 'True Positive Rate (TPR)', angle: -90, position: 'insideLeft', fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '8px', fontSize: '11px', fontWeight: 'bold' }}
                />
                <Area type="monotone" dataKey="tpr" stroke="#4f46e5" strokeWidth={2} fill="#4f46e5" fillOpacity={0.12} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Multi-model performance charts */}
        <div className="lg:col-span-2 p-8 bg-white border border-slate-200 rounded-lg shadow-sm space-y-8">
          <h3 className="font-bold text-slate-900 border-b border-slate-100 pb-4">
            Comparaison des Modèles
          </h3>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={models.map((model) => ({
                  name: model.model,
                  accuracy: model.metrics.accuracy * 100,
                  f1_score: model.metrics.f1_score * 100,
                }))}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} fontWeight="bold" />
                <YAxis stroke="#94a3b8" domain={[0, 100]} fontSize={10} />
                <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '8px' }} />
                <Bar dataKey="accuracy" fill="#4f46e5" radius={[4, 4, 0, 0]} barSize={40} />
                <Bar dataKey="f1_score" fill="#818cf8" radius={[4, 4, 0, 0]} barSize={40} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg text-slate-700 text-sm">
        Visualisation des performances de classification
      </div>
    </div>
  );
};

export default Results;