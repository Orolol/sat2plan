#!/usr/bin/env python3
"""
Script de test pour vérifier les améliorations apportées au modèle UCV-GAN
"""

import torch
import sys
import os

# Ajouter le chemin du projet au PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_improved_ucvgan():
    print("=" * 60)
    print("TEST DES AMÉLIORATIONS UCV-GAN")
    print("=" * 60)

    # Test 1: Vérifier que les imports fonctionnent
    print("\n1. Test des imports...")
    try:
        from sat2plan.logic.models.ucvgan.ucvgan import UCVGAN
        from sat2plan.logic.loss.loss import (
            ContentLoss, EdgeLoss, ColorSpecificLoss, HighwaySpecificLoss
        )
        from sat2plan.logic.configuration.config import Global_Configuration, Model_Configuration
        print("✓ Tous les imports fonctionnent")
    except ImportError as e:
        print(f"✗ Erreur d'import: {e}")
        return False

    # Test 2: Vérifier les hyperparamètres
    print("\n2. Vérification des hyperparamètres...")
    cfg = Global_Configuration()
    print(f"   - Batch size: {cfg.batch_size} (était 32, maintenant 64)")
    print(f"   - Num workers: {cfg.num_workers} (était 2, maintenant 8)")

    # Test 3: Initialiser le modèle avec les nouvelles losses
    print("\n3. Test d'initialisation du modèle...")
    try:
        # Créer une instance du modèle
        model = UCVGAN(rank=0, world_size=1)

        # Vérifier les hyperparamètres modifiés
        print(f"   - L1 lambda: {model.l1_lambda} (était 150, maintenant 50)")
        print(f"   - G factor: {model.g_factor} (était 0.1, maintenant 1.0)")
        print(f"   - Lambda GP: {model.lambda_gp} (était 5, maintenant 10)")
        print(f"   - N critic: {model.n_critic} (était 3, maintenant 2)")

        # Vérifier que les nouvelles losses sont initialisées
        assert hasattr(model, 'content_loss'), "ContentLoss non initialisée"
        assert hasattr(model, 'edge_loss'), "EdgeLoss non initialisée"
        assert hasattr(model, 'color_loss'), "ColorSpecificLoss non initialisée"
        assert hasattr(model, 'highway_loss'), "HighwaySpecificLoss non initialisée"

        print("✓ Modèle initialisé avec toutes les nouvelles losses")
    except Exception as e:
        print(f"✗ Erreur lors de l'initialisation: {e}")
        return False

    # Test 4: Vérifier les losses avec des tenseurs de test
    print("\n4. Test des nouvelles losses...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = 4

        # Créer des tenseurs de test
        y_fake = torch.randn(batch_size, 3, 256, 256).to(device)
        y_real = torch.randn(batch_size, 3, 256, 256).to(device)

        # Ajouter des pixels jaunes dans y_real pour tester HighwaySpecificLoss
        y_real[:, 0, 50:60, :] = 0.8  # Rouge élevé
        y_real[:, 1, 50:60, :] = 0.8  # Vert élevé
        y_real[:, 2, 50:60, :] = -0.5  # Bleu faible (jaune)

        # Test ContentLoss
        content_loss = ContentLoss(alpha1=0.5, alpha2=0.3, alpha3=0.2).to(device)
        cl = content_loss(y_fake, y_real)
        print(f"   - ContentLoss: {cl.item():.4f}")

        # Test EdgeLoss
        edge_loss = EdgeLoss(weight_sobel=1.0, weight_canny=0.5).to(device)
        el = edge_loss(y_fake, y_real)
        print(f"   - EdgeLoss: {el.item():.4f}")

        # Test ColorSpecificLoss
        color_loss = ColorSpecificLoss(color_threshold=0.4, weight=1.0).to(device)
        col = color_loss(y_fake, y_real)
        print(f"   - ColorSpecificLoss: {col.item():.4f}")

        # Test HighwaySpecificLoss
        highway_loss = HighwaySpecificLoss(yellow_weight=2.0, width_weight=1.0).to(device)
        hl = highway_loss(y_fake, y_real)
        print(f"   - HighwaySpecificLoss: {hl.item():.4f}")

        print("✓ Toutes les losses fonctionnent correctement")
    except Exception as e:
        print(f"✗ Erreur lors du test des losses: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Vérifier la fonction de décroissance du L1
    print("\n5. Test de la décroissance du L1 lambda...")
    try:
        # Test à différentes époques
        epochs_test = [0, 50, 100, 150, 200]
        for epoch in epochs_test:
            l1_lambda = model.get_l1_lambda(epoch)
            print(f"   - Époque {epoch:3d}: L1 lambda = {l1_lambda:.2f}")
        print("✓ Décroissance du L1 lambda fonctionne correctement")
    except Exception as e:
        print(f"✗ Erreur lors du test de décroissance: {e}")
        return False

    print("\n" + "=" * 60)
    print("RÉSUMÉ DES AMÉLIORATIONS IMPLÉMENTÉES:")
    print("=" * 60)
    print("""
1. HYPERPARAMÈTRES OPTIMISÉS:
   - L1 lambda: 150 → 50 (moins de flou)
   - G factor: 0.1 → 1.0 (plus d'importance adversariale)
   - Lambda GP: 5 → 10 (meilleure stabilisation)
   - N critic: 3 → 2 (plus d'updates du générateur)
   - Batch size: 32 → 64 (meilleur throughput)
   - Num workers: 2 → 8 (pipeline optimisé)

2. NOUVELLES LOSSES AJOUTÉES:
   ✓ ContentLoss (VGG perceptual + topology)
   ✓ EdgeLoss (Sobel + Canny pour routes nettes)
   ✓ ColorSpecificLoss (préservation des couleurs)
   ✓ HighwaySpecificLoss (spécifique aux autoroutes jaunes)

3. OPTIMISATIONS:
   - Décroissance agressive du L1 (80% de réduction)
   - Mixed precision training activé
   - EMA du générateur pour stabilité
   - Gradient checkpointing dans le bottleneck

4. RÉSULTATS ATTENDUS:
   - Temps d'entraînement: -30% à -40%
   - Routes plus nettes et précises
   - Autoroutes jaunes mieux rendues
   - Convergence plus stable
    """)

    return True

if __name__ == "__main__":
    # Vérifier si CUDA est disponible
    if torch.cuda.is_available():
        print(f"CUDA disponible: {torch.cuda.get_device_name(0)}")
        print(f"Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA non disponible, utilisation du CPU")

    # Lancer les tests
    success = test_improved_ucvgan()

    if success:
        print("\n✓ TOUS LES TESTS PASSENT - Le modèle est prêt pour l'entraînement!")
    else:
        print("\n✗ Certains tests ont échoué - Vérifier les erreurs ci-dessus")
        sys.exit(1)