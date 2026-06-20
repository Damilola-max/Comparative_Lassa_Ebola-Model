# Enhanced PCA with ggplot2
library(ggplot2)
library(dplyr)

BASE <- "/Users/user/CascadeProjects/Comparative_Lassa_Ebola-Model"
df <- read.csv(file.path(BASE, "results/gp_revision/pca_coords.csv"))
df$Virus <- factor(df$Virus, levels = c("Lassa", "Ebola"))

var_exp <- as.numeric(readLines(file.path(BASE, "results/gp_revision/pca_variance.txt")))

centroids <- df %>% group_by(Virus) %>% summarise(PC1 = mean(PC1), PC2 = mean(PC2), .groups = "drop")

colors <- c("Lassa" = "#1565c0", "Ebola" = "#c62828")

p <- ggplot(df, aes(x = PC1, y = PC2, color = Virus, fill = Virus)) +
  geom_point(alpha = 0.12, size = 1.0, shape = 16) +
  stat_ellipse(aes(linetype = Virus), geom = "path", level = 0.95, linewidth = 1.0, alpha = 0.9) +
  geom_density_2d(aes(linetype = Virus), linewidth = 0.6, bins = 5, alpha = 0.6) +
  geom_point(data = centroids, aes(shape = Virus), size = 4, stroke = 1.0, fill = "white") +
  scale_color_manual(values = colors) +
  scale_fill_manual(values = colors) +
  scale_shape_manual(values = c("Lassa" = 21, "Ebola" = 21)) +
  scale_linetype_manual(values = c("Lassa" = "solid", "Ebola" = "solid")) +
  labs(
    title = "ESM-2 Embedding PCA — GP Sequences (n=2,499)",
    x = paste0("PC1 (", var_exp[1], "%)"),
    y = paste0("PC2 (", var_exp[2], "%)"),
    color = "Virus", shape = "Virus", linetype = "Virus"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "gray80", fill = NA, linewidth = 0.5),
    legend.position = c(0.98, 0.98),
    legend.justification = c("right", "top"),
    legend.background = element_rect(fill = "white", color = "gray70"),
    legend.box = "vertical",
    legend.margin = margin(4, 4, 4, 4),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10, color = "gray30")
  )

ggsave(file.path(BASE, "manuscript/assets/refined3_1/media/image6.png"),
       plot = p, width = 8, height = 5.5, dpi = 300, bg = "white")
ggsave(file.path(BASE, "manuscript/assets/refined3_1/media/image6.svg"),
       plot = p, width = 8, height = 5.5, bg = "white")
cat("PCA figure saved.\n")
