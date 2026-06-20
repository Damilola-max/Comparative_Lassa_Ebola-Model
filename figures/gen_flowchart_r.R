# Simple clean vertical pipeline flowchart — no headings, just connected boxes
BASE <- "/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model"
OUT_PNG <- file.path(BASE, "manuscript/assets/refined3_1/media/image1.png")
OUT_SVG <- file.path(BASE, "manuscript/assets/refined3_1/media/image1.svg")

# Colors: soft fills, darker borders
colors <- list(
  i = list(fill = "#E3F2FD", border = "#1565C0"),   # blue
  p = list(fill = "#FFF3E0", border = "#EF6C00"),   # orange
  f = list(fill = "#E8F5E9", border = "#2E7D32"),   # green
  t = list(fill = "#F3E5F5", border = "#7B1FA2"),   # purple
  v = list(fill = "#E0F7FA", border = "#00838F"),   # cyan
  r = list(fill = "#FFEBEE", border = "#C62828"),   # red
  o = list(fill = "#E8EAF6", border = "#283593")    # indigo
)

CX <- 5
W_MAIN <- 4.8
H <- 0.75
FS <- 0.85

# Simple rounded box
draw_box <- function(cx, cy, w, h, text, key, cex = FS, bold = FALSE) {
  xl <- cx - w/2; xr <- cx + w/2
  yb <- cy - h/2; yt <- cy + h/2
  r <- 0.12
  
  # Rounded rectangle polygon
  segs <- 8
  theta <- seq(0, pi/2, length.out = segs)
  
  corner <- function(cx0, cy0, ang_start, ang_end) {
    th <- seq(ang_start, ang_end, length.out = segs)
    list(x = cx0 + r * cos(th), y = cy0 + r * sin(th))
  }
  
  bl <- corner(xl + r, yb + r, pi, 3*pi/2)
  br <- corner(xr - r, yb + r, 3*pi/2, 2*pi)
  tr <- corner(xr - r, yt - r, 0, pi/2)
  tl <- corner(xl + r, yt - r, pi/2, pi)
  
  xs <- c(bl$x, br$x, tr$x, tl$x)
  ys <- c(bl$y, br$y, tr$y, tl$y)
  
  polygon(xs, ys, col = colors[[key]]$fill, border = colors[[key]]$border, lwd = 2.2)
  
  font <- if (bold) 2 else 1
  text(cx, cy, labels = text, cex = cex, font = font, col = "#1a1a1a", linespacing = 1.15)
}

draw_arrow <- function(x1, y1, x2, y2, col = "#424242", lwd = 1.4) {
  arrows(x1, y1, x2, y2, length = 0.09, angle = 25, col = col, lwd = lwd)
}

png(OUT_PNG, width = 7, height = 12, units = "in", res = 300, bg = "white")
svg(OUT_SVG, width = 7, height = 12, bg = "white")
par(mar = c(0, 0, 0, 0), xpd = TRUE)
plot(0, 0, xlim = c(0, 10), ylim = c(-0.5, 13), type = "n", axes = FALSE, xlab = "", ylab = "")

# Title only at top
text(CX, 12.4, "ESM-embedR Pipeline", cex = 1.5, font = 2, col = "#1565C0")
text(CX, 12.05, "From raw sequence to narrative interpretation", cex = 0.85, font = 3, col = "#546E7A")

# Main vertical flow (left column ~x=3.5)
ML <- 3.5   # main left
steps_main <- list(
  list(y = 11.3, key = "i", bold = TRUE,  text = "Raw Sequence\n(FASTA or plain text)"),
  list(y = 10.3, key = "p", bold = TRUE,  text = "clean_sequence()\nuppercase + regex filter"),
  list(y =  9.3, key = "p", bold = FALSE, text = "Whitelist 20 AA codes  |  Regex filter  |  Min length ≥10"),
  list(y =  8.3, key = "p", bold = TRUE,  text = "Cleaned Canonical Sequence"),
  list(y =  7.3, key = "f", bold = TRUE,  text = "amino_acid_frequency_features()\n21-D vector per sequence"),
  list(y =  6.3, key = "f", bold = FALSE, text = "seq_length (1)  +  aa_freq_A…Y (20)"),
  list(y =  5.3, key = "f", bold = TRUE,  text = "Feature Vector: [seq_length, aa_freq_*]"),
  list(y =  4.3, key = "t", bold = TRUE,  text = "LogisticRegression + StandardScaler  |  RandomForest"),
  list(y =  3.3, key = "t", bold = TRUE,  text = "F1-score selection → best model"),
  list(y =  2.3, key = "t", bold = FALSE, text = "Serialize: joblib.dump() → best_model.joblib"),
  list(y =  1.3, key = "t", bold = TRUE,  text = "_build_risk_calibration()")
)

for (i in seq_along(steps_main)) {
  s <- steps_main[[i]]
  draw_box(ML, s$y, W_MAIN, H, s$text, s$key, FS, s$bold)
  if (i > 1) {
    prev <- steps_main[[i-1]]
    draw_arrow(ML, prev$y - H/2, ML, s$y + H/2)
  }
}

# Inference branch (right column ~x=7.5)
IR <- 7.5
steps_inf <- list(
  list(y = 4.3, key = "v", bold = TRUE,  text = "predict_sequences()\nload model"),
  list(y = 3.3, key = "v", bold = FALSE, text = "StandardScaler\n(training params)"),
  list(y = 2.3, key = "v", bold = FALSE, text = "predict_proba\n→ EBOV probability"),
  list(y = 1.3, key = "v", bold = FALSE, text = "Class assignment\nthreshold ≥ 0.5")
)
for (i in seq_along(steps_inf)) {
  s <- steps_inf[[i]]
  draw_box(IR, s$y, 3.0, H, s$text, s$key, FS*0.9, s$bold)
  if (i > 1) {
    prev <- steps_inf[[i-1]]
    draw_arrow(IR, prev$y - H/2, IR, s$y + H/2)
  }
}

# Arrow from Feature Vector to Inference
# Arrow from main flow to inference branch
last_main <- steps_main[[7]]  # Feature Vector
first_inf <- steps_inf[[1]]   # predict_sequences
draw_arrow(ML + W_MAIN/2, last_main$y, IR - 3.0/2, first_inf$y)

# Risk scoring (below inference)
draw_box(IR, 0.3, 3.0, H, "_compute_risk_scores()", "r", FS*0.9, TRUE)
draw_arrow(IR, first_inf$y - H/2 - 0.5, IR, 0.3 + H/2)

# Calibration data arrow from training to risk scoring
draw_arrow(ML + 0.3, 1.3, IR - 0.3, 0.3, col = "#7B1FA2", lwd = 1.2)
text((ML + IR)/2 + 0.3, 0.8, "calibration data", cex = 0.65, col = "#7B1FA2", font = 3)

# Output (below main flow)
draw_box(ML, -0.5, W_MAIN, H, "predicted_virus | confidence | risk_score | narrative", "o", FS*0.85, FALSE)
draw_arrow(ML, 1.3 - H/2, ML, -0.5 + H/2)

# Also arrow from risk scoring to output
draw_arrow(IR - 3.0/2, 0.3, ML + W_MAIN/2, -0.5)

# Footer
text(CX, -1.1, "Deployment: app.py (Streamlit ≥1.30.0)", cex = 0.85, font = 2, col = "#1565C0")
text(CX, -1.4, "https://mutation-analysis.streamlit.app", cex = 0.75, col = "#546E7A")

dev.off()
cat("Clean flowchart saved.\n")
