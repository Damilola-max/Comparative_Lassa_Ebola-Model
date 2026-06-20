# Clean vertical pipeline flowchart matching reference style
# Simple rectangular boxes, phase labels on left, no headings inside flow

BASE <- "/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model"
OUT_PNG <- file.path(BASE, "manuscript/assets/refined3_1/media/image1.png")
OUT_SVG <- file.path(BASE, "manuscript/assets/refined3_1/media/image1.svg")

# Colors by phase
phase_colors <- list(
  p1 = list(fill = "#E3F2FD", border = "#1565C0", label = "#1565C0"),   # blue
  p2 = list(fill = "#FFF3E0", border = "#EF6C00", label = "#EF6C00"),   # orange
  p3 = list(fill = "#E8F5E9", border = "#2E7D32", label = "#2E7D32"),   # green
  p4 = list(fill = "#F3E5F5", border = "#7B1FA2", label = "#7B1FA2"),   # purple
  p5 = list(fill = "#E0F7FA", border = "#00838F", label = "#00838F"),   # cyan
  p6 = list(fill = "#FFEBEE", border = "#C62828", label = "#C62828"),   # red
  p7 = list(fill = "#E8EAF6", border = "#283593", label = "#283593")    # indigo
)

CX <- 5.0
W <- 5.5
H <- 0.85
FS <- 0.82
FS_PHASE <- 0.75

# Simple box with slight rounding
draw_box <- function(cx, cy, w, h, text, key, cex = FS, bold = FALSE) {
  xl <- cx - w/2
  xr <- cx + w/2
  yb <- cy - h/2
  yt <- cy + h/2
  r <- 0.15
  
  # Build rounded rectangle manually
  segs <- 6
  corner <- function(cx0, cy0, a1, a2) {
    th <- seq(a1, a2, length.out = segs)
    list(x = cx0 + r * cos(th), y = cy0 + r * sin(th))
  }
  
  bl <- corner(xl + r, yb + r, pi, 3*pi/2)
  br <- corner(xr - r, yb + r, 3*pi/2, 2*pi)
  tr <- corner(xr - r, yt - r, 0, pi/2)
  tl <- corner(xl + r, yt - r, pi/2, pi)
  
  xs <- c(bl$x, br$x, tr$x, tl$x)
  ys <- c(bl$y, br$y, tr$y, tl$y)
  
  polygon(xs, ys, col = phase_colors[[key]]$fill, 
          border = phase_colors[[key]]$border, lwd = 2.2)
  
  font <- if (bold) 2 else 1
  text(cx, cy, labels = text, cex = cex, font = font, col = "#1a1a1a")
}

draw_arrow <- function(x1, y1, x2, y2, col = "#424242", lwd = 1.5) {
  arrows(x1, y1, x2, y2, length = 0.1, angle = 25, col = col, lwd = lwd)
}

# Phase bracket on left
draw_phase_label <- function(y1, y2, label, color, lx = 1.2) {
  # Draw bracket line
  lines(c(lx, lx), c(y1, y2), col = color, lwd = 2)
  lines(c(lx, lx + 0.15), c(y1, y1), col = color, lwd = 2)
  lines(c(lx, lx + 0.15), c(y2, y2), col = color, lwd = 2)
  # Label rotated
  mid <- (y1 + y2) / 2
  text(lx - 0.15, mid, labels = label, srt = 90, cex = FS_PHASE, 
       font = 2, col = color)
}

png(OUT_PNG, width = 8, height = 13, units = "in", res = 300, bg = "white")
svg(OUT_SVG, width = 8, height = 13, bg = "white")
par(mar = c(0, 0, 0, 0), xpd = TRUE)
plot(0, 0, xlim = c(0, 10), ylim = c(-0.5, 13.5), type = "n", axes = FALSE, xlab = "", ylab = "")

# Title
text(CX, 13.0, "ESM-embedR Pipeline", cex = 1.6, font = 2, col = "#1565C0")
text(CX, 12.65, "From raw sequence to narrative interpretation", cex = 0.85, font = 3, col = "#546E7A")

# Phase 1: Input
draw_phase_label(12.1, 11.3, "PHASE 1  Input", phase_colors$p1$label)
draw_box(CX, 11.7, W, H, "Raw Sequence (FASTA or plain text)", "p1", FS, TRUE)

# Phase 2: Preprocessing
draw_phase_label(10.9, 9.0, "PHASE 2  Preprocessing", phase_colors$p2$label)
draw_box(CX, 10.6, W, H, "clean_sequence()\nuppercase + regex filter", "p2", FS, TRUE)
draw_arrow(CX, 10.6 - H/2, CX, 10.0 + H/2)
draw_box(CX, 10.0, W, H, "Whitelist 20 AA codes  |  Regex filter  |  Min length >= 10", "p2", FS*0.9)
draw_arrow(CX, 10.0 - H/2, CX, 9.4 + H/2)
draw_box(CX, 9.4, W, H, "Cleaned Canonical Sequence", "p2", FS, TRUE)

# Phase 3: Feature Extraction
draw_phase_label(8.3, 7.0, "PHASE 3  Feature Extraction", phase_colors$p3$label)
draw_arrow(CX, 9.4 - H/2, CX, 8.7 + H/2)
draw_box(CX, 8.7, W, H, "amino_acid_frequency_features()\n21-D vector per sequence", "p3", FS, TRUE)
draw_arrow(CX, 8.7 - H/2, CX, 8.1 + H/2)
draw_box(CX, 8.1, W, H, "seq_length (1 feature)  +  aa_freq_A...Y (20 features)", "p3", FS*0.9)
draw_arrow(CX, 8.1 - H/2, CX, 7.5 + H/2)
draw_box(CX, 7.5, W, H, "Feature Vector: [seq_length, aa_freq_*]", "p3", FS, TRUE)

# Phase 4: Training
draw_phase_label(6.8, 4.7, "PHASE 4  Training", phase_colors$p4$label)
draw_arrow(CX, 7.5 - H/2, CX, 6.9 + H/2)
draw_box(CX, 6.9, W, H, "LogisticRegression + StandardScaler  |  RandomForest", "p4", FS*0.9, TRUE)
draw_arrow(CX, 6.9 - H/2, CX, 6.3 + H/2)
draw_box(CX, 6.3, W, H, "F1-score selection -> best model", "p4", FS, TRUE)
draw_arrow(CX, 6.3 - H/2, CX, 5.7 + H/2)
draw_box(CX, 5.7, W, H, "Serialize: joblib.dump() -> best_model.joblib", "p4", FS*0.9)
draw_arrow(CX, 5.7 - H/2, CX, 5.1 + H/2)
draw_box(CX, 5.1, W, H, "_build_risk_calibration()", "p4", FS, TRUE)

# Phase 5: Inference (side branch to the right, then merges back)
# Main continues to output
# Inference branch starts from Feature Vector and goes to the right
IR <- 8.8

# Inference phase label
draw_phase_label(6.9, 4.7, "PHASE 5  Inference", phase_colors$p5$label, lx = 8.8 + W/2 + 0.6)

# Arrow from Feature Vector to inference start
draw_arrow(CX + W/2, 7.5, IR - W/2 + 0.5, 6.3)

draw_box(IR, 6.3, 3.2, H, "predict_sequences()\nload model", "p5", FS*0.85, TRUE)
draw_arrow(IR, 6.3 - H/2, IR, 5.7 + H/2)
draw_box(IR, 5.7, 3.2, H, "StandardScaler\n(training params)", "p5", FS*0.85)
draw_arrow(IR, 5.7 - H/2, IR, 5.1 + H/2)
draw_box(IR, 5.1, 3.2, H, "predict_proba\n-> EBOV probability", "p5", FS*0.85)
draw_arrow(IR, 5.1 - H/2, IR, 4.5 + H/2)
draw_box(IR, 4.5, 3.2, H, "Class assignment\nthreshold >= 0.5", "p5", FS*0.85)

# Risk scoring (phase 6)
draw_phase_label(4.3, 3.7, "PHASE 6  Risk Scoring", phase_colors$p6$label)
draw_arrow(CX, 5.1 - H/2, CX, 4.5 + H/2)
draw_box(CX, 4.5, W, H, "_compute_risk_scores()", "p6", FS, TRUE)

# Arrow from inference to risk scoring
draw_arrow(IR - 3.2/2, 4.5, CX + W/2, 4.5)

# Calibration data dashed arrow
arrows(CX + 0.5, 4.5, IR - 0.5, 4.5, length = 0.07, angle = 25, 
       col = "#7B1FA2", lwd = 1.2, lty = 2)
text((CX + IR)/2, 4.75, "calibration data", cex = 0.6, col = "#7B1FA2", font = 3)

# Phase 7: Output
draw_phase_label(3.5, 2.7, "PHASE 7  Output", phase_colors$p7$label)
draw_arrow(CX, 4.5 - H/2, CX, 3.8 + H/2)
draw_box(CX, 3.8, W, H, "predicted_virus  |  confidence  |  risk_score  |  narrative", "p7", FS*0.85)

# Footer
text(CX, 2.8, "Deployment: app.py (Streamlit >= 1.30.0)", cex = 0.85, font = 2, col = "#1565C0")
text(CX, 2.45, "https://mutation-analysis.streamlit.app", cex = 0.75, col = "#546E7A")

dev.off()
cat("Clean flowchart saved to", OUT_PNG, "and", OUT_SVG, "\n")
