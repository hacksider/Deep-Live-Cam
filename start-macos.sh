#!/bin/bash
# Deep-Live-Cam - Quick Start Script for macOS
# Optimized for Apple Silicon (M1/M2/M3)

echo "🎬 Deep-Live-Cam - Inicialização Rápida"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "✅ Virtual environment found"
    source venv/bin/activate
fi

# Run diagnostic first
echo ""
echo "🔍 Executando diagnóstico..."
python diagnostic.py

echo ""
echo "========================================"
echo "🚀 Iniciando Deep-Live-Cam"
echo "========================================"
echo ""
echo "Opções de inicialização:"
echo ""
echo "1. 🍎 CoreML (Recomendado - Mais rápido)"
echo "2. 💻 CPU (Fallback - Mais lento)"
echo "3. ⚙️  Customizado"
echo ""
read -p "Escolha uma opção (1-3): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Iniciando com CoreML..."
        python run.py --execution-provider coreml
        ;;
    2)
        echo ""
        echo "🚀 Iniciando com CPU..."
        python run.py --execution-provider cpu
        ;;
    3)
        echo ""
        read -p "Execution provider (coreml/cpu): " provider
        python run.py --execution-provider $provider
        ;;
    *)
        echo "Opção inválida. Usando CoreML como padrão."
        python run.py --execution-provider coreml
        ;;
esac
