#!/bin/bash
# Script d'installation des améliorations pour UCV-GAN

echo "=========================================="
echo "Installation des améliorations UCV-GAN"
echo "=========================================="

# Installer kornia pour la détection des bords
echo ""
echo "📦 Installation de kornia..."
pip install kornia>=0.7.0

# Vérifier l'installation
echo ""
echo "✅ Vérification des installations..."
python -c "import kornia; print(f'✓ Kornia version {kornia.__version__} installé avec succès')" || echo "✗ Erreur: Kornia n'a pas pu être installé"

echo ""
echo "=========================================="
echo "Résumé des améliorations implémentées:"
echo "=========================================="
echo ""
echo "1. HYPERPARAMÈTRES OPTIMISÉS:"
echo "   • L1 lambda: 150 → 50 (moins de flou)"
echo "   • G factor: 0.1 → 1.0 (équilibre adversarial)"
echo "   • Lambda GP: 5 → 10 (stabilisation)"
echo "   • N critic: 3 → 2 (plus d'updates générateur)"
echo "   • Batch size: 32 → 64"
echo "   • Num workers: 2 → 8"
echo ""
echo "2. NOUVELLES LOSSES:"
echo "   ✓ ContentLoss (VGG perceptual)"
echo "   ✓ EdgeLoss (Sobel + Laplacian)"
echo "   ✓ ColorSpecificLoss"
echo "   ✓ HighwaySpecificLoss"
echo ""
echo "3. COMMANDE POUR LANCER L'ENTRAÎNEMENT:"
echo "   make run_train_ucvgan"
echo ""
echo "✅ Installation terminée!"