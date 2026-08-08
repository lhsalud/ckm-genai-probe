# =====================================================================
# CKM Ambiguity Probe -- condition summaries + paired comparison of
# mean absolute severity distance (CKM stage known vs. not known)
# =====================================================================

# ---- 0. Paths -------------------------------------------------------
f_stage   <- "ckm_probe_results_full_all.csv"        # use_ckm_stage = TRUE
f_nostage <- "ckm_probe_results_full_no_ckm_stg.csv" # use_ckm_stage = FALSE

# The CSVs were written by pandas, so the boolean columns arrive as the
# strings "True"/"False" and R reads them as character. Coerce them.
read_probe <- function(path) {
  df <- read.csv(path, stringsAsFactors = FALSE)
  bool_cols <- c("use_ehr", "use_rag", "use_ckm_stage",
                 "correct", "parse_ok", "under_triage", "over_triage")
  for (cl in intersect(bool_cols, names(df))) {
    if (is.character(df[[cl]])) df[[cl]] <- toupper(trimws(df[[cl]])) == "TRUE"
  }
  df
}

stage   <- read_probe(f_stage)
nostage <- read_probe(f_nostage)

# ---- 1. Summary function (reproduces the pipeline's JSON block) ------
summarize_condition <- function(df, condition_label) {
  data.frame(
    condition                  = condition_label,
    n                          = nrow(df),
    accuracy                   = round(mean(df$correct), 3),
    under_triage_rate          = round(mean(df$under_triage), 3),
    over_triage_rate           = round(mean(df$over_triage), 3),
    mean_abs_severity_distance = round(mean(abs(df$severity_distance)), 3),
    parse_failure_rate         = round(mean(!df$parse_ok), 3),
    stringsAsFactors           = FALSE
  )
}

summaries <- rbind(
  summarize_condition(stage,   "EHR + RAG + CKM Staging Known"),
  summarize_condition(nostage, "EHR + RAG + CKM Staging Not Known")
)
print(summaries, row.names = FALSE)

# ---- 2. Pair the runs on message_id ---------------------------------
# Both conditions scored the SAME 30 messages, so the two columns of
# severity distances are dependent. Any two-sample test is invalid here.
paired <- merge(
  stage[,   c("message_id", "expected", "predicted", "severity_distance")],
  nostage[, c("message_id", "expected", "predicted", "severity_distance")],
  by = "message_id", suffixes = c("_stage", "_nostage")
)

stopifnot(nrow(paired) == nrow(stage))                       # no unmatched ids
stopifnot(all(paired$expected_stage == paired$expected_nostage))  # same gold labels

paired$abs_stage   <- abs(paired$severity_distance_stage)
paired$abs_nostage <- abs(paired$severity_distance_nostage)
paired$d           <- paired$abs_stage - paired$abs_nostage

cat("\n--- Paired structure ---\n")
cat("n pairs:                ", nrow(paired), "\n")
cat("mean |d| stage-known:   ", round(mean(paired$abs_stage), 3), "\n")
cat("mean |d| stage-unknown: ", round(mean(paired$abs_nostage), 3), "\n")
cat("mean paired difference: ", round(mean(paired$d), 4), "\n")
cat("discordant pairs:       ", sum(paired$d != 0), "of", nrow(paired), "\n\n")
print(table(stage = paired$abs_stage, nostage = paired$abs_nostage))

cat("\nRows where the two conditions disagreed:\n")
print(paired[paired$d != 0,
             c("message_id", "expected_stage",
               "predicted_stage", "predicted_nostage",
               "abs_stage", "abs_nostage")], row.names = FALSE)

# ---- 3a. PRIMARY: exact sign test on discordant pairs ---------------
# |severity_distance| lives on {0,1,2,3} (an ordinal rank difference),
# n = 30, and only a couple of pairs move. Exact, distribution-free.
disc  <- paired$d[paired$d != 0]
k     <- length(disc)
n_pos <- sum(disc > 0)
sign_test <- binom.test(n_pos, k, p = 0.5, alternative = "two.sided")
cat("\n--- Exact sign test (primary) ---\n")
print(sign_test)

# ---- 3b. Exact paired permutation (sign-flip) test on the mean ------
# Enumerates all 2^k sign assignments of the nonzero differences.
perm_signflip <- function(d) {
  nz <- d[d != 0]; k <- length(nz)
  if (k == 0) return(1)
  signs <- as.matrix(expand.grid(rep(list(c(-1, 1)), k)))
  null  <- as.numeric(signs %*% nz) / length(d)
  mean(abs(null) >= abs(mean(d)) - 1e-12)
}
cat("\n--- Exact sign-flip permutation test ---\n")
cat("observed mean difference:", round(mean(paired$d), 4),
    " two-sided p =", round(perm_signflip(paired$d), 4), "\n")

# ---- 3c. Wilcoxon signed-rank (reported for completeness) -----------
# Zeros are dropped and the surviving differences are tied, so R falls
# back to the normal approximation and warns. Prefer 3a/3b.
cat("\n--- Wilcoxon signed-rank (paired) ---\n")
print(suppressWarnings(
  wilcox.test(paired$abs_stage, paired$abs_nostage, paired = TRUE)
))

# ---- 3d. Parametric option: paired t-test ---------------------------
# Treats the SELF_CARE(0) -> ROUTINE(1) -> URGENT(2) -> EMERGENT(3)
# ranks as equally spaced interval data. Shown because it was asked
# for; the equal-spacing assumption is a clinical claim, not a given.
cat("\n--- Paired t-test (parametric, equal-spacing assumption) ---\n")
print(t.test(paired$abs_stage, paired$abs_nostage, paired = TRUE))

# ---- 4. What the design could have detected -------------------------
# With only k discordant pairs, the smallest attainable two-sided exact
# p-value is 2 * 0.5^k -- a floor imposed by the data, not the test.
cat("\nMinimum attainable two-sided exact p with", k, "discordant pairs:",
    min(1, 2 * 0.5^k), "\n")


#########
# Load necessary libraries
library(dplyr)
library(readr)

# 1. Read in the CSV files
df_all <- read_csv("ckm_probe_results_full_all.csv", show_col_types = FALSE)
df_no_stg <- read_csv("ckm_probe_results_full_no_ckm_stg.csv", show_col_types = FALSE)

# 2. Function to compute metrics and print the summary string
generate_summary <- function(df, condition_name) {
  n <- nrow(df)
  accuracy <- mean(df$correct)
  under_triage_rate <- mean(df$under_triage)
  over_triage_rate <- mean(df$over_triage)
  
  # Calculate mean absolute severity distance
  mean_abs_severity_distance <- mean(abs(df$severity_distance))
  
  # Assuming parse_ok is logical (TRUE/FALSE)
  parse_failure_rate <- 1 - mean(df$parse_ok) 
  
  # Print the formatted summary
  cat(sprintf('Summary: { "condition": "%s", "n": %d, "accuracy": %.3f, "under_triage_rate": %.3f, "over_triage_rate": %.3f, "mean_abs_severity_distance": %.3f, "parse_failure_rate": %.1f }\n',
              condition_name, n, accuracy, under_triage_rate, over_triage_rate, mean_abs_severity_distance, parse_failure_rate))
}

# Print the summaries
cat("--- SUMMARIES ---\n")
generate_summary(df_all, "EHR + RAG + CKM Staging Known")
generate_summary(df_no_stg, "EHR + RAG + CKM Staging Not Known")
cat("\n")

# 3. Statistical Testing
# Merge the datasets by message_id to ensure we are comparing paired samples
merged_df <- inner_join(df_all, df_no_stg, by = "message_id", suffix = c("_all", "_no_stg"))

# Extract the absolute severity distances
abs_dist_all <- abs(merged_df$severity_distance_all)
abs_dist_no_stg <- abs(merged_df$severity_distance_no_stg)

cat("--- STATISTICAL SIGNIFICANCE TESTING ---\n")

# Non-Parametric: Wilcoxon Signed-Rank Test (Recommended for ordinal ranks)
cat("\n1. Wilcoxon Signed-Rank Test (Non-Parametric)\n")
wilcox_res <- wilcox.test(abs_dist_all, abs_dist_no_stg, paired = TRUE)
print(wilcox_res)

# Parametric: Paired T-Test
cat("\n2. Paired T-Test (Parametric)\n")
t_test_res <- t.test(abs_dist_all, abs_dist_no_stg, paired = TRUE)
print(t_test_res)

#
# Load necessary libraries
library(dplyr)
library(ggplot2)
library(readr)

# 1. Read in the dataset 
# (You can swap this with the 'no_stg' file to plot the other condition)
df <- read_csv("ckm_probe_results_full_all.csv", show_col_types = FALSE)

# 2. Code the values for the triage levels and calculate severity_distance
df <- df %>%
  mutate(
    # Assign numerical ranks to the predicted labels
    predicted_rank = case_when(
      predicted == "EMERGENT"  ~ 3,
      predicted == "URGENT"    ~ 2,
      predicted == "ROUTINE"   ~ 1,
      predicted == "SELF_CARE" ~ 0,
      TRUE ~ NA_real_ # Catch-all for any parsing errors
    ),
    
    # Assign numerical ranks to the expected (ground truth) labels
    expected_rank = case_when(
      expected == "EMERGENT"  ~ 3,
      expected == "URGENT"    ~ 2,
      expected == "ROUTINE"   ~ 1,
      expected == "SELF_CARE" ~ 0,
      TRUE ~ NA_real_
    ),
    
    # Calculate the severity distance using subtraction
    # Negative values = Under-triage, Positive values = Over-triage, 0 = Correct
    calculated_severity_distance = predicted_rank - expected_rank
  )

# Verify the calculated distances match your original column
# (Optional sanity check)
print(table(df$severity_distance == df$calculated_severity_distance))

# ---------------------------------------------------------
# VISUALIZATION 1: Heatmap of Predicted vs Expected
# ---------------------------------------------------------

# First, convert the text labels into ordered factors so they plot 
# logically from lowest (Self Care) to highest (Emergent) acuity.
level_order <- c("SELF_CARE", "ROUTINE", "URGENT", "EMERGENT")
df$expected_factor <- factor(df$expected, levels = level_order)
df$predicted_factor <- factor(df$predicted, levels = level_order)

# Aggregate the counts for the heatmap
count_data <- df %>%
  count(expected_factor, predicted_factor) %>%
  tidyr::complete(expected_factor, predicted_factor, fill = list(n = 0)) # Fill 0s for empty cells

plot_heatmap <- ggplot(count_data, aes(x = expected_factor, y = predicted_factor, fill = n)) +
  geom_tile(color = "white", size = 1) +
  geom_text(aes(label = n), color = ifelse(count_data$n > (max(count_data$n)/2), "white", "black"), size = 6) +
  scale_fill_gradient(low = "#f0f9e8", high = "#0868ac", name = "Count") +
  labs(
    title = "Actual vs. Predicted Triage Levels",
    subtitle = "EHR + RAG + CKM Staging Known",
    x = "Expected (Actual) Triage Level",
    y = "Predicted Triage Level"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 11),
    axis.text.y = element_text(size = 11),
    plot.title = element_text(face = "bold", size = 14)
  )

# Display the first plot
print(plot_heatmap)


# ---------------------------------------------------------
# VISUALIZATION 2: Distribution of Severity Distances
# ---------------------------------------------------------

plot_distance <- ggplot(df, aes(x = calculated_severity_distance)) +
  geom_bar(fill = "#2b8cbe", color = "black", alpha = 0.8) +
  scale_x_continuous(breaks = -3:3, limits = c(-3.5, 3.5)) +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed", size = 1) +
  labs(
    title = "Distribution of Triage Severity Errors",
    subtitle = "Negative = Under-triage | Positive = Over-triage | 0 = Correct Match",
    x = "Severity Distance (Predicted - Expected)",
    y = "Number of Cases"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.title.x = element_text(margin = margin(t = 10))
  )

# Display the second plot
print(plot_distance)

###
# Load necessary libraries
library(dplyr)
library(readr)

# 1. Read in both CSV files
df_all <- read_csv("ckm_probe_results_full_all.csv", show_col_types = FALSE)
df_no_stg <- read_csv("ckm_probe_results_full_no_ckm_stg.csv", show_col_types = FALSE)

# 2. Add a column to identify which experimental condition the row belongs to
df_all <- df_all %>% mutate(condition = "EHR + RAG + CKM Staging Known")
df_no_stg <- df_no_stg %>% mutate(condition = "EHR + RAG + CKM Staging Not Known")

# 3. Combine both datasets into one
df_combined <- bind_rows(df_all, df_no_stg)

# 4. Code the factors (ranks) and calculate severity distance for the entire combined dataset
df_combined <- df_combined %>%
  mutate(
    # Assign numerical ranks to the predicted labels
    predicted_rank = case_when(
      predicted == "EMERGENT"  ~ 3,
      predicted == "URGENT"    ~ 2,
      predicted == "ROUTINE"   ~ 1,
      predicted == "SELF_CARE" ~ 0,
      TRUE ~ NA_real_
    ),
    
    # Assign numerical ranks to the expected labels
    expected_rank = case_when(
      expected == "EMERGENT"  ~ 3,
      expected == "URGENT"    ~ 2,
      expected == "ROUTINE"   ~ 1,
      expected == "SELF_CARE" ~ 0,
      TRUE ~ NA_real_
    ),
    
    # Calculate the severity distance using subtraction
    calculated_severity_distance = predicted_rank - expected_rank
  )

# 5. Save the combined and coded dataset to a new CSV
write_csv(df_combined, "ckm_probe_results_combined_coded.csv")

# Print a quick summary to the console to verify
cat("Successfully combined and coded datasets.\n")
cat("Total rows written:", nrow(df_combined), "\n")
print(table(df_combined$condition))
