import React, { useState, useEffect } from 'react';
import { Save, Play, ChevronRight } from 'lucide-react';
import { trainModel, tuneModel } from '../services/api';
import { useNavigate, useSearchParams } from 'react-router-dom';

const Training = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const selectedAlgos = searchParams.get('algos') ? searchParams.get('algos').split(',') : [];
  const initialTrainType = searchParams.get('trainType') || 'scratch';
  
  const [params, setParams] = useState({
    learningRate: 0.01,
    maxDepth: 10,
    nEstimators: 100,
    k: 5
  });
  const [trainType, setTrainType] = useState(initialTrainType);
  const [savedConfigName, setSavedConfigName] = useState('');
  const [isTraining, setIsTraining] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem('ml_training_config');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (parsed.params) setParams(parsed.params);
        if (parsed.trainType) setTrainType(parsed.trainType);
      } catch {
        // ignore invalid saved config
      }
    }
  }, []);

  const handleParamChange = (name, value) => {
    setParams(prev => ({ ...prev, [name]: value }));
  };

  const handleTrain = async () => {
    try {
      setIsTraining(true);
      const trainingParams = {
        ...params,
        algorithms: selectedAlgos.length > 0 ? selectedAlgos : ['KNN', 'SVM', 'Random Forest', 'Logistic Regression'],
        trainType,
      };
      await trainModel(trainingParams);
      alert('Entraînement démarré en arrière-plan. Vous pouvez suivre l\'avancée dans Historique.');
      navigate('/history');
    } catch (error) {
      console.error(error);
      alert('Erreur lors du lancement de l\'entraînement');
    } finally {
      setIsTraining(false);
    }
  };

  const handleSave = () => {
    localStorage.setItem('ml_training_config', JSON.stringify({ trainType, params }));
    setSavedConfigName('Configuration sauvegardée');
    setTimeout(() => setSavedConfigName(''), 2400);
  };

  const handleAutoTune = async (method) => {
    if (selectedAlgos.length === 0) {
      alert('Sélectionnez au moins un algorithme pour lancer le tuning.');
      return;
    }
    try {
      setIsTraining(true);
      await tuneModel({ algorithms: selectedAlgos, method });
      alert(`Tuning automatique (${method}) démarré. Consultez Historique pour suivre l\'avancée.`);
      navigate('/history');
    } catch (error) {
      console.error(error);
      alert('Erreur lors du démarrage du tuning auto.');
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="space-y-8">
      <header className="flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Entraînement</h1>
        </div>
        <div className="flex flex-wrap gap-3">
          <button
            onClick={handleSave}
            className="flex items-center gap-2 px-5 py-2 bg-slate-600 hover:bg-slate-700 text-white font-bold rounded-lg transition-all"
          >
            <Save className="w-4 h-4" />
            Sauvegarder
          </button>
          <button 
            onClick={handleTrain}
            disabled={isTraining}
            className={`flex items-center gap-2 px-6 py-2 text-white font-bold rounded-lg transition-all ${isTraining ? 'bg-blue-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'}`}
          >
            <Play className="w-4 h-4 fill-current" />
            {isTraining ? 'Démarrage...' : 'Lancer'}
          </button>
        </div>
      </header>

      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        {selectedAlgos.length > 0 && (
          <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm font-bold text-blue-700">
              Algorithmes: {selectedAlgos.join(', ')}
            </p>
          </div>
        )}
        <div className="flex gap-2 items-center">
          <span className="text-sm font-semibold text-slate-600">Mode :</span>
          <button
            onClick={() => setTrainType('scratch')}
            className={`px-3 py-2 rounded-lg text-sm font-bold ${trainType === 'scratch' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-700'}`}
          >From Scratch</button>
          <button
            onClick={() => setTrainType('pretrained')}
            className={`px-3 py-2 rounded-lg text-sm font-bold ${trainType === 'pretrained' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-700'}`}
          >Pré-entraîné</button>
        </div>
      </div>

      {savedConfigName && (
        <div className="rounded-lg bg-green-50 border border-green-100 p-4 text-sm text-green-700">
          {savedConfigName}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Manual Configuration */}
        <div className="lg:col-span-8 space-y-6">
          <div className="p-6 bg-white border border-slate-200 rounded-lg shadow-sm space-y-6">
            <h3 className="text-lg font-bold text-slate-900 border-b border-slate-100 pb-4">
              Hyperparamètres
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Learning Rate */}
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-bold text-slate-700">Learning Rate</label>
                  <span className="text-xs text-blue-600 font-bold">{params.learningRate}</span>
                </div>
                <input
                  type="range" min="0.0001" max="0.1" step="0.0001"
                  value={params.learningRate}
                  onChange={(e) => handleParamChange('learningRate', parseFloat(e.target.value))}
                  className="w-full accent-blue-600 h-1.5 bg-slate-100 rounded-lg"
                />
              </div>

              {/* Max Depth */}
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-bold text-slate-700">Max Depth</label>
                  <span className="text-xs text-blue-600 font-bold">{params.maxDepth}</span>
                </div>
                <input
                  type="range" min="1" max="50" step="1"
                  value={params.maxDepth}
                  onChange={(e) => handleParamChange('maxDepth', parseInt(e.target.value))}
                  className="w-full accent-blue-600 h-1.5 bg-slate-100 rounded-lg"
                />
              </div>

              {/* Number of Trees */}
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-bold text-slate-700">Nombre d'arbres</label>
                  <span className="text-xs text-blue-600 font-bold">{params.nEstimators}</span>
                </div>
                <input
                  type="range" min="10" max="500" step="10"
                  value={params.nEstimators}
                  onChange={(e) => handleParamChange('nEstimators', parseInt(e.target.value))}
                  className="w-full accent-blue-600 h-1.5 bg-slate-100 rounded-lg"
                />
              </div>

              {/* k (KNN) */}
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-bold text-slate-700">Valeur de k</label>
                  <span className="text-xs text-blue-600 font-bold">{params.k}</span>
                </div>
                <input
                  type="range" min="1" max="25" step="1"
                  value={params.k}
                  onChange={(e) => handleParamChange('k', parseInt(e.target.value))}
                  className="w-full accent-blue-600 h-1.5 bg-slate-100 rounded-lg"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Automatic Tuning */}
        <div className="lg:col-span-4 space-y-6">
          <div className="p-6 bg-white border border-slate-200 rounded-lg shadow-sm space-y-6">
            <h3 className="text-lg font-bold text-slate-900">Tuning Automatique</h3>

            <div className="space-y-2">
              <button
                onClick={() => handleAutoTune('GridSearch')}
                className="w-full text-left p-3 bg-slate-50 border border-slate-100 rounded-lg hover:bg-blue-50 text-sm font-bold text-slate-700 transition-all"
              >
                GridSearch
              </button>

              <button
                onClick={() => handleAutoTune('RandomSearch')}
                className="w-full text-left p-3 bg-slate-50 border border-slate-100 rounded-lg hover:bg-blue-50 text-sm font-bold text-slate-700 transition-all"
              >
                RandomSearch
              </button>

              <button
                onClick={() => handleAutoTune('Optuna')}
                className="w-full text-left p-3 bg-slate-50 border border-slate-100 rounded-lg hover:bg-blue-50 text-sm font-bold text-slate-700 transition-all"
              >
                Optuna Tuning
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Training;