# 🚀 Guia Rápido - Deep-Live-Cam Otimizado para macOS

## ⚡ Início Rápido

### 1️⃣ Executar com Otimizações

```bash
# Opção A: Script automático (Recomendado)
./start-macos.sh

# Opção B: Comando direto
python run.py --execution-provider coreml
```

### 2️⃣ Verificar Performance

```bash
# Verificar se otimizações estão aplicadas
python diagnostic.py

# Testar velocidade de processamento
python benchmark.py
```

---

## 🎯 Configurações para LIVE SWAP

### 🏆 Máximo FPS (Recomendado)

Na interface do Deep-Live-Cam:

```
✅ Manter estas opções:
- [ ] Keep fps: OFF
- [ ] Keep audio: OFF (não usado no live)
- [ ] Keep frames: OFF
- [ ] Many faces: OFF
- [ ] Map faces: OFF
- [ ] Poisson Blend: OFF
- [ ] Fix Blueish Cam: OFF
- [x] Show FPS: ON (para monitorar)
- [ ] Mouth Mask: OFF

❌ Desativar TODOS os enhancers:
- [ ] Face Enhancer: OFF
- [ ] GPEN Enhancer 256: OFF
- [ ] GPEN Enhancer 512: OFF
```

**FPS Esperado:** 10-15 FPS

---

### ⚖️ Qualidade Média (Bom para Live)

```
✅ Ativar:
- [x] Show FPS: ON
- [x] Face Enhancer: ON

❌ Manter desativado:
- [ ] GPEN Enhancer 256: OFF
- [ ] GPEN Enhancer 512: OFF

**FPS Esperado:** 6-10 FPS
```

---

### 🎨 Máxima Qualidade (Não recomendado para Live)

```
⚠️ Apenas para testes - FPS muito baixo!

✅ Ativar:
- [x] Face Enhancer: ON
- [x] GPEN Enhancer 256: ON
- [ ] GPEN Enhancer 512: OFF (muito lento)

**FPS Esperado:** 3-5 FPS
```

---

## 📊 Monitorando FPS

1. Ative **"Show FPS"** na interface
2. O FPS aparecerá no canto superior esquerdo do preview
3. **FPS Ideal para Live:** > 8 FPS
4. **FPS Mínimo aceitável:** > 5 FPS

---

## 🔧 Solução de Problemas

### ❌ FPS < 4

**Causa:** Processamento muito lento

**Solução:**
1. Desative todos os enhancers
2. Feche outros aplicativos (Chrome, Slack)
3. Verifique se está usando CoreML:
   ```bash
   python run.py --execution-provider coreml
   ```

### ❌ "No face detected"

**Causa:** Face não detectada na imagem de origem

**Solução:**
1. Use uma foto frontal bem iluminada
2. Rosto deve estar visível (sem óculos escuros, máscara)
3. Aproxime o rosto da câmera
4. Melhore a iluminação do ambiente

### ❌ "Camera failed"

**Causa:** Permissão de câmera negada

**Solução:**
1. Abra **System Settings**
2. Vá em **Privacy & Security > Camera**
3. Permita **Terminal** (ou iTerm/VSCode)
4. Reinicie o Terminal e tente novamente

### ❌ Tela cinza/azul

**Causa:** Câmera errada selecionada

**Solução:**
1. Clique em **"Refresh"** no seletor de câmera
2. Selecione outra câmera da lista
3. Evite "Capture screen" ou dispositivos virtuais

---

## 💡 Dicas de Performance

### 1. Use Fonte de Qualidade

- Foto frontal bem iluminada
- Rosto centralizado
- Sem óculos ou acessórios
- Resolução média (500x500 a 1000x1000)

### 2. Otimizações Manuais (Avançado)

Editar `modules/ui.py`:

```python
# Reduzir resolução para mais FPS (mínimo: 320x180)
PREVIEW_DEFAULT_WIDTH = 480   # Was 640
PREVIEW_DEFAULT_HEIGHT = 270  # Was 360
```

### 3. Fechar Outros Apps

- Chrome: ~2GB RAM
- Slack: ~500MB RAM
- VSCode: ~1GB RAM

**Economia:** Até 3.5GB RAM = mais recursos para o DLC

### 4. Usar External Camera

Webcams externas geralmente têm:
- Melhor qualidade de imagem
- Menos latência
- Melhor detecção facial

---

## 📈 Performance Esperada

### MacBook M1 (8GB RAM)

```
Configuração Máximo FPS: 10-15 FPS
Configuração Qualidade: 6-10 FPS
Configuração Extrema: 2-4 FPS
```

### MacBook M1 Pro/Max (16GB+ RAM)

```
Configuração Máximo FPS: 12-18 FPS
Configuração Qualidade: 8-12 FPS
Configuração Extrema: 3-5 FPS
```

### MacBook M2/M3 (8GB+ RAM)

```
Configuração Máximo FPS: 12-20 FPS
Configuração Qualidade: 8-14 FPS
Configuração Extrema: 4-6 FPS
```

---

## 🎬 Usando no OBS Studio

1. **Inicie o Deep-Live-Cam** em modo Live
2. **Abra o OBS Studio**
3. **Adicione nova fonte:**
   - Tipo: **Window Capture**
   - Janela: **Preview** (do Deep-Live-Cam)
4. **Crop da janela:**
   - Segure `Alt` e arraste as bordas
   - Deixe apenas o preview
5. **Configure output:**
   - Resolution: 640x360 ou 1280x720
   - FPS: 30 ou 60

---

## 📞 Suporte

### Logs e Diagnóstico

```bash
# Salvar diagnóstico em arquivo
python diagnostic.py > diagnostic-output.txt

# Enviar para suporte
- diagnostic-output.txt
- Screenshot do FPS
- Configuração usada
```

### Informações Úteis

Ao pedir ajuda, inclua:

```
1. Modelo do MacBook: (ex: M1 Pro 14" 2021)
2. macOS version: (ex: 14.2 Sonoma)
3. FPS atual: (ex: 6 FPS)
4. Configuração: (ex: CoreML, enhancers off)
5. Output do diagnostic.py
```

---

## ✅ Checklist Antes de Usar

- [ ] Virtual environment ativado
- [ ] Dependencies instaladas (`pip install -r requirements.txt`)
- [ ] Modelos na pasta `models/`
- [ ] Permissão de câmera concedida
- [ ] CoreML disponível (`python diagnostic.py`)
- [ ] Enhancers desativados (para máximo FPS)
- [ ] Show FPS ativado (para monitorar)

---

## 🎯 Comandos Úteis

```bash
# Ativar virtual environment
source venv/bin/activate

# Verificar otimizações
python diagnostic.py

# Testar performance
python benchmark.py

# Iniciar com CoreML
python run.py --execution-provider coreml

# Iniciar com CPU (fallback)
python run.py --execution-provider cpu

# Iniciar em modo headless (CLI)
python run.py -s source.jpg -t target.mp4 -o output.mp4
```

---

**Versão:** 2.0.5c-optimized  
**Última atualização:** 2026-02-26  
**Platforma:** macOS Apple Silicon (M1/M2/M3)
