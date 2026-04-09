import React, { useState } from 'react';
import { ChevronRight, Info, CheckCircle2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const ALGORITHMS = [
  {
    id: 'svm',
    name: 'SVM',
    fullName: 'Support Vector Machine',
    description: 'Un modèle robuste pour les problèmes de classification binaire.',
    docs: 'Le SVM identifie une frontière de séparation optimale entre les classes de films selon leurs caractéristiques.',
  },
  {
    id: 'rf',
    name: 'Random Forest',
    fullName: 'Random Forest',
    description: 'Stable et précis sur des données hétérogènes.',
    docs: 'La forêt aléatoire agrège plusieurs arbres de décision pour réduire le surapprentissage et améliorer la stabilité.',
  },
  {
    id: 'knn',
    name: 'KNN',
    fullName: 'K-Nearest Neighbors',
    description: 'Classe un film en se basant sur ses voisins les plus proches.',
    docs: 'KNN utilise les films similaires en budget, durée et popularité pour prédire le succès.',
  },
  {
    id: 'lr',
    name: 'Logistic Regression',
    fullName: 'Régression Logistique',
    description: 'Simple, interprétable, efficace pour des prédictions binaires.',
    docs: 'La régression logistique calcule la probabilité qu’un film devienne un hit à partir de ses caractéristiques.',
  },
];

const ModelCard = ({ algo, isSelected, onToggle, openDocs }) => (
  <div
    className={`rounded-3xl border p-6 transition-all duration-200 ${isSelected ? 'border-indigo-500 bg-indigo-50' : 'border-slate-200 bg-white hover:border-indigo-300 hover:bg-slate-50'}`}
  >
    <div className="flex items-start justify-between gap-4">
      <div>
        <p className="text-xs uppercase tracking-[0.3em] text-slate-500">{algo.fullName}</p>
        <h3 className="mt-3 text-2xl font-semibold text-slate-900">{algo.name}</h3>
      </div>
      <button
        type="button"
        onClick={() => openDocs(algo)}
        className="rounded-full border border-slate-200 p-2 text-slate-500 hover:bg-slate-100"
      >
        <Info className="w-4 h-4" />
      </button>
    </div>
    <p className="mt-5 text-sm leading-7 text-slate-600">{algo.description}</p>
    <button
      type="button"
      onClick={() => onToggle(algo.id)}
      className={`mt-6 w-full rounded-full px-4 py-3 text-sm font-semibold transition ${isSelected ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-900 hover:bg-slate-200'}`}
    >
      {isSelected ? 'Sélectionné' : 'Choisir'}
    </button>
  </div>
);

const Home = () => {
  const navigate = useNavigate();
  const [selectedModels, setSelectedModels] = useState([]);
  const [trainType, setTrainType] = useState('scratch');
  const [activeDocs, setActiveDocs] = useState(null);

  const toggleModel = (id) => {
    setSelectedModels((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const handleContinue = () => {
    if (selectedModels.length === 0) return;
    navigate(`/training?algos=${selectedModels.join(',')}&trainType=${trainType}`);
  };

  return (
    <div className="space-y-10">
      <header className="space-y-4">
        <div className="max-w-3xl">
          <p className="text-sm uppercase tracking-[0.3em] text-slate-500">Accueil</p>
          <h1 className="mt-4 text-4xl font-extrabold text-slate-900">Choisissez votre modèle de prédiction</h1>
          <p className="mt-4 text-base leading-7 text-slate-600">Sélectionnez un ou plusieurs algorithmes, puis lancez l’entraînement pour estimer le succès d’un film.</p>
        </div>
      </header>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {ALGORITHMS.map((algo) => (
          <ModelCard
            key={algo.id}
            algo={algo}
            isSelected={selectedModels.includes(algo.id)}
            onToggle={toggleModel}
            openDocs={setActiveDocs}
          />
        ))}
      </div>

      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div className="rounded-3xl border border-slate-200 bg-white p-6">
         
        </div>

        <button
          type="button"
          disabled={selectedModels.length === 0}
          onClick={handleContinue}
          className={`inline-flex items-center justify-center gap-2 rounded-full px-6 py-3 text-sm font-semibold transition ${selectedModels.length === 0 ? 'bg-slate-200 text-slate-500 cursor-not-allowed' : 'bg-indigo-600 text-white hover:bg-indigo-700'}`}
        >
          {selectedModels.length === 0 ? 'Sélectionnez un modèle' : 'Continuer'}
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>

      {activeDocs && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-6">
          <div className="w-full max-w-2xl rounded-3xl bg-white p-8 shadow-2xl">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm uppercase tracking-[0.3em] text-indigo-600">Documentation</p>
                <h2 className="mt-3 text-3xl font-bold text-slate-900">{activeDocs.fullName}</h2>
              </div>
              <button
                type="button"
                onClick={() => setActiveDocs(null)}
                className="rounded-full border border-slate-200 p-3 text-slate-500 hover:bg-slate-100"
              >
                ✕
              </button>
            </div>
            <p className="mt-6 text-slate-600 leading-7">{activeDocs.docs}</p>
            <button
              type="button"
              onClick={() => setActiveDocs(null)}
              className="mt-10 inline-flex items-center gap-2 rounded-full bg-indigo-600 px-5 py-3 text-sm font-semibold text-white hover:bg-indigo-700"
            >
              <CheckCircle2 className="w-4 h-4" />
              Fermer
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Home;
