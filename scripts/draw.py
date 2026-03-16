import graphviz

# Create the main graph
dot = graphviz.Digraph('UAV_Detector', comment='UAV Detection Architecture')
dot.attr(rankdir='TB')  # Top to Bottom

# Define node styles
dot.attr('node', shape='box', style='filled', color='black', fillcolor='white', fontname='Arial')

# Main Pipeline Nodes
dot.node('input', 'Input UAV Image')
dot.node('backbone', 'YOLOv8n Backbone')
dot.node('neck', 'FPN/PAN Neck')

# Feature Map Cluster
with dot.subgraph(name='cluster_pyramid') as c:
    c.attr(label='(Added P2 scale)', labelloc='b')
    c.node('p2', 'P2\n(added)')
    c.node('p3', 'P3')
    c.node('p4', 'P4')
    c.node('p5', 'P5')
    c.attr(color='white') # Make the cluster boundary invisible

# Lower Pipeline Nodes
dot.node('predictions', 'Detection Predictions')
dot.node('losses', 'Detection Losses')

# Training Path Nodes
dot.node('assignment', 'Assignment Module\n(TAL or Area-aware TAL)')
dot.node('gt_boxes', 'GT Boxes')
dot.node('gt_computation', 'GT Area Computation')
dot.node('positive_assignment', 'Positive Assignment')
dot.node('area_weight', 'Area Weight')

# Set training only label for assignment module
# dot.edge('gt_boxes', 'assignment', label='GT Boxes', labelloc='t')

# Define edge styles
dot.attr('edge', color='black', fontname='Arial', fontsize='10')

# -- Main Pipeline Edges --
dot.edge('input', 'backbone')
dot.edge('backbone', 'neck')

# Branch to pyramid scales
dot.edge('neck', 'p2')
dot.edge('neck', 'p3')
dot.edge('neck', 'p4')
dot.edge('neck', 'p5')

# Consolidate from pyramid scales
dot.edge('p2', 'predictions')
dot.edge('p3', 'predictions')
dot.edge('p4', 'predictions')
dot.edge('p5', 'predictions')

dot.edge('predictions', 'losses')

# -- Training Path Edges --
# Dashed arrow with label
dot.edge('predictions', 'assignment', style='dashed', label='(training only)')

dot.edge('gt_boxes', 'assignment')
dot.edge('gt_boxes', 'gt_computation')
dot.edge('gt_computation', 'area_weight')
dot.edge('assignment', 'positive_assignment')
dot.edge('positive_assignment', 'losses')

# Complex edge: Arrow from assignment splits to positive assignment, but line from assignment connects to area weight line
dot.edge('area_weight', 'positive_assignment', constraint='false', weight='1', minlen='1', dir='none', arrowhead='none')
dot.edge('assignment', 'positive_assignment', lhead='positive_assignment', minlen='1', tailport='s', headport='n')
# Need to manually make a horizontal connector or manage layout, graphviz is tough with this specific connection.
# Alternative simplified connection:
dot.edge('area_weight', 'positive_assignment', label='Area Weight', labelloc='t', constraint='false')

# Adjust layout to match image.
# For example, align gt computation below gt boxes. This is naturally handled by rankdir='TB'.
# We may need to use subgraph clusters or rank=same to force side-by-side alignment.
with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('gt_boxes')
    s.node('assignment')
    # This might not align perfectly and may require fine-tuning with node placement or different layout engines like 'neato' but is the standard way with dot.

print(dot.source)
# Save and render
dot.render('uav_detection_diagram', format='png', cleanup=False)