import React, { useState } from 'react';
import { Play } from 'lucide-react';
import { predict } from '../services/api';

const Prediction = () => {
    const [formData, setFormData] = useState({
        budget: 50000000,
        runtime: 120,
        release_year: 2024,
        genres: 'Action, Adventure',
        cast: 'Tom Cruise',
        crew: 'Steven Spielberg',
        popularity: 50,
        vote_average: 7.0,
        vote_count: 1000
    });

    const [isPredicting, setIsPredicting] = useState(false);
    const [prediction, setPrediction] = useState(null);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handlePredict = async (e) => {
        e.preventDefault();
        setIsPredicting(true);
        try {
            const response = await predict(formData);
            setPrediction(response.data.prediction);
        } catch (error) {
            console.error("Error predicting:", error);
            setPrediction("Erreur de prédiction");
        } finally {
            setIsPredicting(false);
        }
    };

    return (
        <div className="space-y-8">
            <header className="bg-blue-600 p-8 rounded-lg text-white">
                <h1 className="text-4xl font-bold">Prédiction</h1>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                {/* Form Section */}
                <div className="lg:col-span-8 bg-white p-6 rounded-lg shadow-sm border border-slate-200">
                    <h3 className="text-xl font-bold text-slate-900 mb-6">
                        Paramètres du Film
                    </h3>

                    <form onSubmit={handlePredict} className="space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {/* Budget */}
                            <div className="space-y-2">
                                <label className="text-sm font-bold text-slate-700">
                                    Budget ($)
                                </label>
                                <input
                                    type="number" name="budget"
                                    value={formData.budget} onChange={handleChange}
                                    className="w-full px-4 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                                    required
                                />
                            </div>

                            {/* Runtime */}
                            <div className="space-y-2">
                                <label className="text-sm font-bold text-slate-700">
                                    Durée (minutes)
                                </label>
                                <input
                                    type="number" name="runtime"
                                    value={formData.runtime} onChange={handleChange}
                                    className="w-full px-4 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500"
                                    required
                                />
                            </div>

                            {/* Release Year */}
                            <div className="space-y-2">
                                <label className="text-sm font-bold text-slate-700">
                                    Année
                                </label>
                                <input
                                    type="number" name="release_year"
                                    value={formData.release_year} onChange={handleChange}
                                    className="w-full px-4 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500"
                                    required
                                />
                            </div>

                            {/* Genres */}
                            <div className="space-y-2">
                                <label className="text-sm font-bold text-slate-700">
                                    Genres
                                </label>
                                <input
                                    type="text" name="genres" placeholder="Sci-Fi, Action..."
                                    value={formData.genres} onChange={handleChange}
                                    className="w-full px-4 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500"
                                />
                            </div>

                            {/* Cast */}
                            <div className="space-y-2">
                                <label className="text-sm font-bold text-slate-700">
                                    Acteurs
                                </label>
                                <input
                                    type="text" name="cast" placeholder="Leonardo DiCaprio, Brad Pitt..."
                                    value={formData.cast} onChange={handleChange}
                                    className="w-full px-4 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500"
                                />
                            </div>

                            {/* Crew/Director */}
                            <div className="space-y-2">
                                <label className="text-sm font-bold text-slate-700">
                                    Réalisateur
                                </label>
                                <input
                                    type="text" name="crew" placeholder="Christopher Nolan"
                                    value={formData.crew} onChange={handleChange}
                                    className="w-full px-4 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500"
                                />
                            </div>

                            {/* Popularity */}
                            <div className="space-y-2">
                                <label className="text-sm font-bold text-slate-700">
                                    Popularité
                                </label>
                                <input
                                    type="number" name="popularity" step="0.1"
                                    value={formData.popularity} onChange={handleChange}
                                    className="w-full px-4 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500"
                                    required
                                />
                            </div>

                            {/* Vote Average */}
                            <div className="space-y-2">
                                <label className="text-sm font-bold text-slate-700">
                                    Note moyenne (/10)
                                </label>
                                <input
                                    type="number" name="vote_average" step="0.1" min="0" max="10"
                                    value={formData.vote_average} onChange={handleChange}
                                    className="w-full px-4 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500"
                                    required
                                />
                            </div>

                            {/* Vote Count */}
                            <div className="space-y-2">
                                <label className="text-sm font-bold text-slate-700">
                                    Nombre de votes
                                </label>
                                <input
                                    type="number" name="vote_count"
                                    value={formData.vote_count} onChange={handleChange}
                                    className="w-full px-4 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:ring-2 focus:ring-blue-500"
                                    required
                                />
                            </div>

                        </div>

                        <div className="pt-4">
                            <button
                                type="submit"
                                disabled={isPredicting}
                                className={`w-full flex items-center justify-center gap-3 py-3 rounded-lg text-white font-bold transition-all ${isPredicting ? 'bg-blue-400 cursor-wait' : 'bg-blue-600 hover:bg-blue-700'}`}
                            >
                                {isPredicting ? (
                                    <span>Analyse...</span>
                                ) : (
                                    <>
                                        <Play className="w-4 h-4 fill-current" />
                                        Prédire
                                    </>
                                )}
                            </button>
                        </div>
                    </form>
                </div>

                {/* Results Section */}
                <div className="lg:col-span-4">
                    <div className={`p-6 rounded-lg border transition-all ${prediction ? 'bg-green-50 border-green-300' : 'bg-slate-50 border-slate-200'}`}>
                        <h3 className={`text-sm font-bold uppercase mb-4 ${prediction ? 'text-green-700' : 'text-slate-500'}`}>
                            Résultat
                        </h3>

                        {prediction ? (
                            <div className="space-y-2">
                                <p className="text-sm text-slate-600">Prédiction</p>
                                <p className={`text-5xl font-bold ${prediction === 'Hit' ? 'text-green-600' : 'text-red-600'}`}>
                                    {prediction}
                                </p>
                            </div>
                        ) : (
                            <div className="py-8 flex flex-col items-center justify-center text-center text-slate-500">
                                <p className="text-sm font-medium">Remplissez le formulaire et cliquez sur prédire.</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Prediction;
