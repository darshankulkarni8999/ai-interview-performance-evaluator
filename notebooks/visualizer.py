import os
import json
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# LOAD SESSION RESULTS
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_dir, '..', 'outputs', 'session_results.json')

with open(results_path, 'r') as f:
    data = json.load(f)

summary = data['summary']
detailed = data['detailed_results']

print(f"Loaded {len(detailed)} question results.")
print(f"Average Overall Score: {summary['avg_overall_score']}/100\n")


# ============================================================
# CHART 1: Score Breakdown per Question (Grouped Bar Chart)
# ============================================================

def plot_score_breakdown(detailed, output_path):
    """
    For each question, shows relevance, completeness, and overall
    score as grouped bars side by side — easy to compare across
    questions at a glance.
    """
    labels = [f"Q{i+1}\n[{r['skill']}]" for i, r in enumerate(detailed)]
    relevance_scores = [r['relevance_score'] for r in detailed]
    completeness_scores = [r['completeness_score'] for r in detailed]
    overall_scores = [r['overall_score'] for r in detailed]

    x = np.arange(len(labels))   # positions for each question on x-axis
    width = 0.25                  # width of each bar

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot three bars per question, offset using width
    bars1 = ax.bar(x - width, relevance_scores, width, label='Relevance', color='#4C72B0')
    bars2 = ax.bar(x, completeness_scores, width, label='Completeness', color='#DD8452')
    bars3 = ax.bar(x + width, overall_scores, width, label='Overall', color='#55A868')

    # Add value labels on top of each bar
    for bar in bars1 + bars2 + bars3:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Questions')
    ax.set_ylabel('Score (out of 100)')
    ax.set_title('Score Breakdown per Question')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 115)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# CHART 2: Overall Score Trend (Line Chart)
# ============================================================

def plot_score_trend(detailed, output_path):
    """
    Shows overall score as a line across all questions —
    helps visualize if performance improved or declined
    as the session progressed.
    """
    question_nums = [f"Q{i+1}" for i in range(len(detailed))]
    overall_scores = [r['overall_score'] for r in detailed]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(question_nums, overall_scores,
            marker='o', color='#4C72B0', linewidth=2, markersize=8)

    # Add score labels next to each point
    for i, score in enumerate(overall_scores):
        ax.annotate(f'{score}',
                    xy=(i, score),
                    xytext=(0, 10), textcoords="offset points",
                    ha='center', fontsize=9)

    # Add horizontal reference lines for performance zones
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Excellent (80+)')
    ax.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Good (60+)')
    ax.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='Needs work (40+)')

    ax.set_xlabel('Question')
    ax.set_ylabel('Overall Score')
    ax.set_title('Overall Score Trend Across Session')
    ax.set_ylim(0, 115)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# CHART 3: Summary Radar-style Bar Chart
# ============================================================

def plot_summary_chart(summary, output_path):
    """
    Shows a single bar chart of the average scores across
    all four metrics — a clean summary visualization for
    the README or report.
    """
    metrics = ['Avg Relevance', 'Avg Completeness', 'Avg Overall']
    values = [
        summary['avg_relevance'],
        summary['avg_completeness'],
        summary['avg_overall_score']
    ]

    colors = ['#4C72B0', '#DD8452', '#55A868']

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(metrics, values, color=colors, width=0.4)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', fontsize=11, fontweight='bold')

    ax.set_ylabel('Score (out of 100)')
    ax.set_title('Session Performance Summary')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# CHART 4: Filler Word Usage
# ============================================================

def plot_filler_words(detailed, output_path):
    """
    Shows total filler words used per question as a bar chart.
    Helps candidates visually see where they used the most
    filler words during the session.
    """
    labels = [f"Q{i+1}" for i in range(len(detailed))]
    filler_counts = [r['filler_word_count'] for r in detailed]

    colors = ['#d9534f' if c > 2 else '#f0ad4e' if c > 0 else '#5cb85c'
              for c in filler_counts]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, filler_counts, color=colors)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', fontsize=10)

    ax.set_xlabel('Question')
    ax.set_ylabel('Filler Word Count')
    ax.set_title('Filler Word Usage per Question')
    ax.set_ylim(0, max(filler_counts) + 3 if filler_counts else 5)
    ax.grid(axis='y', alpha=0.3)

    # Color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#5cb85c', label='None (0)'),
        Patch(facecolor='#f0ad4e', label='Moderate (1-2)'),
        Patch(facecolor='#d9534f', label='High (3+)')
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# RUN ALL CHARTS
# ============================================================

outputs_dir = os.path.join(script_dir, '..', 'outputs')

plot_score_breakdown(detailed,
    os.path.join(outputs_dir, 'chart_score_breakdown.png'))

plot_score_trend(detailed,
    os.path.join(outputs_dir, 'chart_score_trend.png'))

plot_summary_chart(summary,
    os.path.join(outputs_dir, 'chart_summary.png'))

plot_filler_words(detailed,
    os.path.join(outputs_dir, 'chart_filler_words.png'))

print("\nAll charts saved to outputs/ folder.")
print("Open the outputs/ folder to view your visualizations.")