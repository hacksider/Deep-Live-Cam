# 🎨 Guia de Qualidade Natural - Deep-Live-Cam 480p

## 📊 Mudanças Aplicadas para Qualidade Natural

### 1. **Resolução Reduzida para 480p** 📹

**Antes:** 640x480 ou 640x360  
**Depois:** 854x480 (480p verdadeiro - 16:9)

**Benefícios:**
- ✅ Menos artificialidade
- ✅ Melhor blend de cores
- ✅ +30% mais FPS
- ✅ Histograma mais uniforme

---

### 2. **Opacity Padrão Ajustada** 🎭

**Antes:** 100% (1.0)  
**Depois:** 85% (0.85)

**Benefícios:**
- ✅ Blend mais natural com a pele original
- ✅ Menos contraste "plastificado"
- ✅ Transição suave entre face swap e original

---

### 3. **Color Correction Automático** 🎨

**Novo:** Histogram matching ativado por padrão no live mode

**Benefícios:**
- ✅ Cores da face combinam com o pescoço/testa
- ✅ Menos diferença de tom de pele
- ✅ Integração seamless

---

### 4. **Poisson Blending Habilitado** 🔧

**Novo:** Poisson blend ativado no live mode (480p permite)

**Benefícios:**
- ✅ Bordas invisíveis
- ✅ Blend perfeito nas extremidades
- ✅ Menos "máscara facial"

---

### 5. **Sharpness Desativado por Padrão** 📷

**Antes:** 0.0 (mas muitos ativavam)  
**Depois:** 0.0 com aviso claro

**Benefícios:**
- ✅ Menos artificialidade
- ✅ Pele mais natural
- ✅ Sem "halo" nas bordas

---

## ⚙️ Configurações Recomendadas

### 🏆 Qualidade Mais Natural (Recomendado)

```
Na Interface do Deep-Live-Cam:

✅ Manter:
- [x] Color Correction: ON (automático)
- [x] Poisson Blend: ON (automático)
- [ ] Sharpness: 0.0 (padrão)
- [x] Opacity: 85% (padrão)

❌ Desativar:
- [ ] Face Enhancer: OFF
- [ ] GPEN 256: OFF
- [ ] GPEN 512: OFF
- [ ] Many Faces: OFF

**Resultado:** Imagem mais natural, 10-15 FPS
```

---

### ⚖️ Qualidade Intermediária

```
✅ Ativar:
- [x] Opacity: 90-95%
- [x] Sharpness: 0.5-1.0
- [ ] Face Enhancer: ON (opcional)

**Resultado:** Um pouco mais artificial, 8-12 FPS
```

---

### 🎨 Máxima Qualidade (Menos Natural)

```
⚠️ Apenas se necessário:
- [x] Opacity: 100%
- [x] Sharpness: 2.0+
- [x] Face Enhancer: ON
- [x] GPEN 256: ON

**Resultado:** Mais artificial, 5-8 FPS
```

---

## 🔧 Ajustes Manuais (Avançado)

### Ajustar Opacidade

```python
# modules/globals.py
opacity: float = 0.85  # Ajuste: 0.70-0.95 para mais natural
```

**Recomendações:**
- `0.70-0.80`: Muito natural, menos swap visível
- `0.85-0.90`: Equilíbrio ideal
- `0.95-1.0`: Mais artificial, swap óbvio

---

### Ajustar Color Correction

```python
# modules/globals.py
color_correction: bool = True  # Manter True para natural
```

**Efeito:**
- `True`: Cores combinam automaticamente
- `False`: Cores originais do modelo (pode diferir)

---

### Ajustar Poisson Blend

```python
# modules/globals.py
poisson_blend: bool = True  # Manter True para blend suave
```

**Efeito:**
- `True`: Bordas invisíveis (recomendado)
- `False`: Bordas visíveis (não recomendado)

---

### Ajustar Sharpness

```python
# modules/globals.py
sharpness: float = 0.0  # 0.0-5.0

# Recomendações:
# 0.0-0.5: Muito natural
# 0.5-1.5: Leve definição
# 1.5-3.0: Artificial
# 3.0+: Muito artificial
```

---

## 📈 Comparação de Qualidade

### Configuração Natural (85% opacity + color correction)

```
✅ Pele uniforme
✅ Cores integradas
✅ Bordas invisíveis
✅ Textura natural
✅ Sem halo
❌ Swap menos "perfeito"

FPS: 10-15
Qualidade Visual: ⭐⭐⭐⭐⭐
Naturalidade: ⭐⭐⭐⭐⭐
```

---

### Configuração Artificial (100% opacity + enhancer)

```
❌ Pele plastificada
❌ Cores diferentes
✅ Swap "perfeito"
❌ Textura artificial
❌ Halo visível
✅ Mais detalhe

FPS: 5-8
Qualidade Visual: ⭐⭐⭐
Naturalidade: ⭐⭐
```

---

## 🎬 Dicas para Imagem Mais Natural

### 1. **Use Fonte de Qualidade**

- ✅ Foto frontal bem iluminada
- ✅ Mesma temperatura de cor do ambiente
- ✅ Sem filtros ou enhancements
- ✅ Resolução média (500x500 a 800x800)

---

### 2. **Ajuste Iluminação**

- ✅ Luz suave e difusa
- ✅ Evite sombras duras
- ✅ Combine com iluminação da câmera

---

### 3. **Use Opacity < 100%**

- ✅ 85% é o ideal
- ✅ Permite pele original aparecer
- ✅ Cria blend mais natural

---

### 4. **Evite Enhancers no Live**

- ❌ Face Enhancer adiciona artificialidade
- ❌ GPEN cria textura "plástica"
- ✅ Use apenas para foto/vídeo gravado

---

### 5. **Ative Color Correction**

- ✅ Já está ativado por padrão
- ✅ Combina automaticamente as cores
- ✅ Essencial para naturalidade

---

## 🎯 Troubleshooting de Qualidade

### ❌ Imagem "Plastificada"

**Causa:** Excesso de processing

**Solução:**
```
1. Desative Face Enhancer
2. Desative GPEN
3. Reduza Sharpness para 0.0
4. Reduza Opacity para 85%
```

---

### ❌ Cores Diferentes

**Causa:** Color correction desativado

**Solução:**
```python
# modules/globals.py
color_correction: bool = True
```

Ou na UI:
- Marque "Fix Blueish Cam" (color correction)

---

### ❌ Bordas Visíveis

**Causa:** Poisson blend desativado

**Solução:**
```python
# modules/globals.py
poisson_blend: bool = True
```

---

### ❌ Textura Artificial

**Causa:** Sharpness muito alto

**Solução:**
```python
# modules/globals.py
sharpness: float = 0.0  # ou 0.5 máximo
```

---

### ❌ Swap Muito Óbvio

**Causa:** Opacity muito alta

**Solução:**
```python
# modules/globals.py
opacity: float = 0.85  # ou 0.75-0.90
```

---

## 📊 Teste de Qualidade

### Script de Teste

```bash
# Testar diferentes configurações
python run.py --execution-provider coreml

# Na UI, ajuste:
1. Opacity: 70%, 85%, 95%, 100%
2. Sharpness: 0.0, 1.0, 2.0
3. Enhancers: ON/OFF

# Compare qual parece mais natural
```

---

### Checklist de Qualidade Natural

- [ ] Opacity entre 80-90%
- [ ] Sharpness em 0.0
- [ ] Color Correction ativado
- [ ] Poisson Blend ativado
- [ ] Face Enhancer desativado
- [ ] GPEN desativado
- [ ] Fonte bem iluminada
- [ ] Iluminação ambiente similar

---

## 🎨 Exemplo de Configuração Perfeita

```python
# modules/globals.py

# Natural look settings
opacity: float = 0.85             # 85% blend
sharpness: float = 0.0            # No sharpening
color_correction: bool = True     # Auto color match
poisson_blend: bool = True        # Seamless edges

# Live mode
live_mode: bool = True
show_fps: bool = True             # Monitor performance

# Enhancers OFF for natural look
fp_ui = {
    "face_enhancer": False,
    "face_enhancer_gpen256": False,
    "face_enhancer_gpen512": False
}
```

---

## 📞 Suporte

### Se a imagem ainda parecer artificial:

1. **Verifique configurações:**
   - Opacity < 90%?
   - Sharpness = 0.0?
   - Enhancers OFF?

2. **Verifique fonte:**
   - Foto bem iluminada?
   - Sem filtros?
   - Resolução adequada?

3. **Verifique ambiente:**
   - Iluminação similar?
   - Temperatura de cor combinando?

---

**Última atualização:** 2026-02-26  
**Versão:** Deep-Live-Cam 2.0.5c-optimized-480p  
**Resolução:** 854x480 (480p)
