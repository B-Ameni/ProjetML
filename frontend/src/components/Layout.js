import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Activity, Settings, Database, History, PieChart } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs) {
    return twMerge(clsx(inputs));
}

const SidebarItem = ({ to, icon: Icon, label }) => (
    <NavLink
        to={to}
        className={({ isActive }) => cn(
            "flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 group",
            isActive
                ? "bg-indigo-50 text-indigo-700 border border-indigo-100"
                : "text-slate-500 hover:bg-slate-50 hover:text-slate-900"
        )}
    >
        <Icon className="w-5 h-5 transition-transform duration-200 group-hover:scale-105" />
        <span className="font-semibold text-sm">{label}</span>
    </NavLink>
);

const Layout = ({ children }) => {
    return (
        <div className="flex min-h-screen bg-slate-50 text-slate-900">
            {/* Sidebar */}
            <aside className="w-64 border-r border-slate-200 bg-white p-6 flex flex-col gap-8 fixed h-full shadow-sm">
                <div className="flex items-center gap-3 px-2">
                    <span className="text-2xl font-black tracking-tighter text-indigo-600 font-display">Box<span className="text-slate-900">Office</span>ML</span>
                </div>

                <nav className="flex flex-col gap-1">
                    <SidebarItem to="/" icon={LayoutDashboard} label="Accueil Models" />
                    <SidebarItem to="/dashboard" icon={Database} label="Gestion du Dataset" />
                    <SidebarItem to="/prediction" icon={Activity} label="Simulateur Box-Office" />
                    <SidebarItem to="/training" icon={Settings} label="Hyperparamètres" />
                    <SidebarItem to="/results" icon={PieChart} label="Résultats" />
                    <SidebarItem to="/history" icon={History} label="Historique MLflow" />
                </nav>

            </aside>

            {/* Main Content */}
            <main className="flex-1 ml-64 min-h-screen">
                <div className="max-w-6xl mx-auto p-10">
                    {children}
                </div>
            </main>
        </div>
    );
};

export default Layout;
