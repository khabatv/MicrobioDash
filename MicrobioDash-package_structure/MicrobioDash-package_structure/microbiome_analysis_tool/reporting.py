
import os
from io import BytesIO
import base64
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from .config import logger

def generate_pdf_report(global_data, interpretations, output_dir):
    try:
        report_path = os.path.join(output_dir, 'microbiome_analysis_report.pdf')
        doc = SimpleDocTemplate(report_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # --- Helper to convert figures to images ---
        def fig_to_image(fig):
            img_buffer = BytesIO()
            fig.write_image(img_buffer, format='png', scale=2)
            img_buffer.seek(0)
            return Image(img_buffer, width=400, height=300)

        elements.append(Paragraph("Microbiome Analysis Report", styles['Title']))
        elements.append(Spacer(1, 12))

        # --- Add plots from global_data ---
        if 'seq_depth_fig' in global_data and global_data['seq_depth_fig']:
            elements.append(Paragraph("Sequencing Depth", styles['Heading2']))
            elements.append(fig_to_image(global_data['seq_depth_fig']))
            elements.append(Spacer(1, 12))

        if 'alpha_fig' in global_data and global_data['alpha_fig']:
            elements.append(Paragraph("Alpha Diversity", styles['Heading2']))
            elements.append(fig_to_image(global_data['alpha_fig']))
            elements.append(Spacer(1, 12))

        if 'pcoa_fig' in global_data and global_data['pcoa_fig']:
            elements.append(Paragraph("Beta Diversity (PCoA)", styles['Heading2']))
            elements.append(fig_to_image(global_data['pcoa_fig']))
            elements.append(Spacer(1, 12))

        if 'abundance_order_plot' in global_data and global_data['abundance_order_plot']:
            elements.append(Paragraph("Relative Abundance by Order", styles['Heading2']))
            elements.append(fig_to_image(global_data['abundance_order_plot']))
            elements.append(Spacer(1, 12))

        if 'tree_img' in global_data and global_data['tree_img']:
            elements.append(Paragraph("Phylogenetic Tree", styles['Heading2']))
            tree_img_data = base64.b64decode(global_data['tree_img'].split(',')[1])
            img_buffer = BytesIO(tree_img_data)
            elements.append(Image(img_buffer, width=400, height=300))
            elements.append(Spacer(1, 12))

        elements.append(Paragraph("AI Interpretations", styles['Heading2']))
        # Clean up markdown for PDF
        interp_text = interpretations.replace('*', '').replace('`', '').replace('\n', '<br/>')
        elements.append(Paragraph(interp_text, styles['BodyText']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Output Files", styles['Heading2']))
        elements.append(Paragraph(f"ASV Table saved to: {os.path.join(output_dir, 'microbiome_ai_16s_asv.csv')}", styles['BodyText']))
        elements.append(Paragraph(f"Taxonomy Table saved to: {os.path.join(output_dir, 'microbiome_ai_taxonomy.csv')}", styles['BodyText']))

        doc.build(elements)
        logger.info(f"PDF report generated successfully at: {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}", exc_info=True)
        return None
