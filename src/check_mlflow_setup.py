"""
Script de vérification de la configuration MLflow Model Registry
Vérifie que tout est bien configuré avant de commencer
"""

import sys
from pathlib import Path
import requests
import time

# Ajouter src au chemin
sys.path.insert(0, str(Path(__file__).parent))

from mlflow_config import BACKEND_STORE_URI, ARTIFACT_ROOT, TRACKING_URI

def check_dependencies():
    """Vérifie que les dépendances sont installées"""
    print("🔍 Vérification des dépendances...")
    
    deps = ['mlflow', 'sklearn', 'pandas', 'fastapi']
    failed = []
    
    for dep in deps:
        try:
            __import__(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep} - MANQUANT")
            failed.append(dep)
    
    if failed:
        print(f"\n⚠️  Installez les dépendances manquantes:")
        print(f"   pip install {' '.join(failed)}")
        return False
    
    return True

def check_directories():
    """Vérifie que les répertoires existent"""
    print("\n🔍 Vérification des répertoires...")
    
    dirs = [
        Path(ARTIFACT_ROOT),
        Path(BACKEND_STORE_URI.replace("sqlite:///", "")).parent,
    ]
    
    for d in dirs:
        if d.exists():
            print(f"   ✅ {d}")
        else:
            print(f"   ⚠️  {d} - Va être créé au démarrage du serveur")
    
    return True

def check_mlflow_server():
    """Vérifie que le serveur MLflow est accessible"""
    print("\n🔍 Vérification du serveur MLflow...")
    
    try:
        response = requests.get(TRACKING_URI, timeout=2)
        print(f"   ✅ Serveur MLflow accessible: {TRACKING_URI}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Serveur MLflow NON accessible: {TRACKING_URI}")
        print(f"\n   💡 Lancez d'abord le serveur avec:")
        print(f"      python src/start_mlflow_server.py")
        return False
    except Exception as e:
        print(f"   ⚠️  Erreur: {e}")
        return False

def check_mlflow_config():
    """Vérifie la configuration MLflow"""
    print("\n🔍 Vérification de la configuration MLflow...")
    
    print(f"   📍 Backend Store: {BACKEND_STORE_URI}")
    print(f"   📍 Artifact Root: {ARTIFACT_ROOT}")
    print(f"   📍 Tracking URI: {TRACKING_URI}")
    
    return True

def main():
    print("=" * 70)
    print("🔧 MLflow Model Registry - Vérification de Configuration")
    print("=" * 70)
    print()
    
    checks = [
        ("Dépendances", check_dependencies),
        ("Répertoires", check_directories),
        ("Configuration", check_mlflow_config),
        ("Serveur MLflow", check_mlflow_server),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Erreur lors de la vérification {name}: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DES VÉRIFICATIONS")
    print("=" * 70)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:.<40} {status}")
    
    all_passed = all(r for _, r in results)
    
    print()
    if all_passed:
        print("🎉 Toutes les vérifications sont passées!")
        print("\n✨ Vous êtes prêt à utiliser le Model Registry!")
        print("\nProchaines étapes:")
        print("   1. Entraîner un modèle: python src/train.py")
        print("   2. Enregistrer le modèle: python src/register_best_model.py")
        print("   3. Accéder à l'interface Web: http://localhost:5000")
        return 0
    else:
        print("⚠️  Certaines vérifications ont échoué")
        print("\nRésolvez les problèmes ci-dessus puis réessayez")
        return 1

if __name__ == "__main__":
    sys.exit(main())
