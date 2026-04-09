import React from "react";
import {
    History,
    RotateCcw,
    Tag,
    BarChart2,
    Bell
} from "lucide-react";

import { getModels, rollback } from "../services/api";


// ==============================
// Experiment Card
// ==============================

const ExperimentCard = ({ exp, onRollback }) => {

    const handleRollback = async () => {
        try {
            await rollback(exp.run_id);
            alert(`Version ${exp.version} (${exp.model}) restaurée comme Déployée.`);
            // Refresh the list
            onRollback();
        } catch (error) {
            console.error("Rollback error:", error);
            alert("Erreur lors du rollback: " + error.message);
        }
    };

    return (
        <div className="p-6 bg-white border border-slate-200 rounded-2xl hover:border-indigo-300 transition-all shadow-sm relative overflow-hidden group">

            {/* Status Badge */}

            <div
                className={`absolute top-0 right-0 text-[10px] font-bold px-3 py-1 rounded-bl-xl
                ${exp.status === "Déployé"
                        ? "bg-emerald-50 text-emerald-600 border-l border-b border-emerald-100"
                        : "bg-slate-50 text-slate-500 border-l border-b border-slate-100"
                    }`}
            >
                {exp.status}
            </div>


            {/* Header */}

            <div className="mb-6">

                <h3 className="text-lg font-bold text-slate-900 mb-1">
                    {exp.name}
                </h3>

                <div className="flex items-center gap-3 text-slate-400 text-[10px] font-bold uppercase tracking-wider">

                    <span className="flex items-center gap-1">
                        <Tag className="w-3 h-3" />
                        {exp.id}
                    </span>

                </div>

            </div>



            {/* Info Grid */}

            <div className="grid grid-cols-2 gap-4 mb-6">

                <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                    <p className="text-[10px] text-slate-400 font-bold uppercase tracking-widest mb-1">
                        Version (MLflow)
                    </p>
                    <p className="text-sm text-indigo-600 font-bold">
                        {exp.version}
                    </p>
                </div>


                <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                    <p className="text-[10px] text-slate-400 font-bold uppercase tracking-widest mb-1">
                        Dataset
                    </p>
                    <p className="text-sm text-slate-700 font-bold">
                        {exp.dataset}
                    </p>
                </div>


                <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                    <p className="text-[10px] text-slate-400 font-bold uppercase tracking-widest mb-1">
                        Algorithme
                    </p>
                    <p className="text-sm text-slate-700 font-bold">
                        {exp.model}
                    </p>
                </div>


                <div className="p-3 bg-indigo-50 rounded-lg border border-indigo-100">
                    <p className="text-[10px] text-indigo-400 font-bold uppercase tracking-widest mb-1">
                        Accuracy
                    </p>
                    <p className="text-sm text-indigo-700 font-bold">
                        {(exp.accuracy * 100).toFixed(1)}%
                    </p>
                </div>

            </div>



            {/* Buttons */}

            <div className="flex gap-2">

                <button
                    onClick={() =>
                        alert(
                            `Comparaison de ${exp.name} avec le modèle déployé... (Fonctionnalité à implémenter)`
                        )
                    }
                    className="flex-1 flex items-center justify-center gap-2 py-2.5 bg-slate-50 hover:bg-slate-100 text-slate-700 text-xs font-bold rounded-xl transition-all border border-slate-200"
                >
                    <BarChart2 className="w-4 h-4 text-slate-400" />
                    Comparer
                </button>


                <button
                    onClick={handleRollback}
                    disabled={exp.status === "Déployé"}
                    className={`flex-1 flex items-center justify-center gap-2 py-2.5 text-xs font-bold rounded-xl transition-all shadow-md ${
                        exp.status === "Déployé"
                            ? "bg-emerald-100 text-emerald-600 cursor-not-allowed opacity-50 border border-emerald-200"
                            : "bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-100 border-0"
                    }`}
                >
                    <RotateCcw className="w-4 h-4" />
                    {exp.status === "Déployé" ? "Déployé" : "Rollback"}
                </button>

            </div>

        </div>
    );
};



// ==============================
// History Page
// ==============================

const HistoryPage = () => {

    const [experiments, setExperiments] = React.useState([]);
    const [loading, setLoading] = React.useState(true);
    const [lastSync, setLastSync] = React.useState(null);



    // ==========================
    // Fetch Models
    // ==========================

    const fetchModels = async () => {
        setLoading(true);
        try {
            const response = await getModels();
            setExperiments(response.data.models);
            setLastSync(new Date());
        } catch (error) {
            console.error("Error fetching models:", error);
        } finally {
            setLoading(false);
        }
    };

    React.useEffect(() => {
        fetchModels();
        // Auto-refresh every 30 seconds instead of 5
        const interval = setInterval(fetchModels, 30000);
        return () => clearInterval(interval);
    }, []);




    return (

        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-2 duration-500">


            {/* Header */}

            <header className="flex justify-between items-center">

                <div>

                    <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
                        Historique
                    </h1>

                    <p className="text-slate-500 mt-1">
                        Historique des expérimentations et gestion des versions.
                    </p>

                </div>


                <div className="flex items-center gap-4">

                    <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-lg">

                        <span className="text-[10px] font-bold text-slate-600 uppercase">
                            Dernier sync: {lastSync ? lastSync.toLocaleTimeString('fr-FR', {hour: '2-digit', minute: '2-digit', second: '2-digit'}) : 'En attente...'}
                        </span>

                    </div>


                    <button
                        onClick={fetchModels}
                        disabled={loading}
                        className="flex items-center gap-2 px-6 py-2.5 bg-white border border-slate-200 text-slate-700 font-bold rounded-xl hover:bg-slate-50 transition-all shadow-sm active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
                    >

                        <History className="w-4 h-4" />

                        Synchroniser

                    </button>

                </div>

            </header>



            {/* Loading */}

            {loading ? (

                <div className="text-center text-slate-400 font-bold py-20">
                    Chargement des modèles...
                </div>

            ) : experiments.length > 0 ? (



                /* Models Grid */

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

                    {experiments.map(exp => (

                        <ExperimentCard
                            key={exp.id}
                            exp={exp}
                            onRollback={fetchModels}
                        />

                    ))}

                </div>

            ) : (



                /* Empty State */

                <div className="p-10 bg-white border-2 border-dashed border-slate-200 rounded-3xl flex flex-col items-center justify-center text-center gap-4">

                    <div className="w-16 h-16 bg-slate-50 rounded-full flex items-center justify-center text-slate-300">

                        <History className="w-8 h-8" />

                    </div>

                    <div>

                        <h4 className="text-xl font-bold text-slate-900">
                            Aucune expérimentation trouvée
                        </h4>

                    </div>

                </div>

            )}

        </div>

    );

};

export default HistoryPage;