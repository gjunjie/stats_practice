"""
Script to convert SUMMARY.md to PDF
"""
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    packages = ['markdown', 'weasyprint']
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def convert_md_to_pdf(md_file='SUMMARY.md', pdf_file='SUMMARY.pdf'):
    """Convert markdown file to PDF"""
    try:
        import markdown
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install required packages: pip install markdown weasyprint")
        return False
    
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    md = markdown.Markdown(extensions=['extra', 'tables', 'codehilite'])
    html_content = md.convert(md_content)
    
    # Add CSS styling for better PDF appearance
    css_style = """
    <style>
        @page {
            size: A4;
            margin: 1.5cm 1.5cm 2cm 1.5cm;
            @bottom-center {
                content: "Page " counter(page) " of " counter(pages);
                font-size: 8pt;
                color: #666;
                font-family: 'Helvetica', 'Arial', sans-serif;
                margin-bottom: 0.5cm;
            }
        }
        body {
            font-family: 'Helvetica', 'Arial', sans-serif;
            font-size: 9pt;
            line-height: 1.2;
            color: #333;
        }
        h1 {
            font-size: 16pt;
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 5px;
            margin-top: 15px;
            margin-bottom: 4px;
        }
        h2 {
            font-size: 13pt;
            color: #34495e;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 3px;
            margin-top: 15px;
            margin-bottom: 3px;
        }
        h3 {
            font-size: 11pt;
            color: #555;
            margin-top: 12px;
            margin-bottom: 2px;
        }
        h4 {
            font-size: 10pt;
            color: #555;
            margin-top: 8px;
            margin-bottom: 2px;
        }
        table {
            border-collapse: collapse;
            width: 90%;
            margin: 3px auto;
            font-size: 8pt;
            border: 1px solid #000;
        }
        th, td {
            border: 0.5px solid #333;
            padding: 4px 6px;
            text-align: left;
            font-size: 8pt;
        }
        th {
            background-color: #2c2c2c;
            color: #fff;
            font-weight: bold;
            text-align: left;
            border-bottom: 1px solid #000;
            padding: 5px 6px;
        }
        td {
            background-color: #fff;
            color: #000;
        }
        tr:nth-child(even) td {
            background-color: #f8f8f8;
        }
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 4px auto;
        }
        img.small-plot {
            max-width: 60%;
            height: auto;
            display: block;
            margin: 3px auto;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 8pt;
        }
        pre {
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 8pt;
        }
        blockquote {
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-left: 0;
            color: #666;
            font-size: 8.5pt;
        }
        p {
            font-size: 9pt;
            margin: 2px 0;
            line-height: 1.2;
        }
        li {
            font-size: 9pt;
            margin: 1px 0;
            line-height: 1.2;
        }
        ul, ol {
            margin: 3px 0;
            padding-left: 20px;
        }
        ul ul, ol ul, ul ol, ol ol {
            margin: 2px 0;
            padding-left: 25px;
        }
    </style>
    """
    
    # Combine HTML
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        {css_style}
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Convert HTML to PDF
    print(f"Converting {md_file} to {pdf_file}...")
    HTML(string=full_html, base_url=os.path.dirname(os.path.abspath(md_file))).write_pdf(pdf_file)
    print(f"Successfully created {pdf_file}!")
    return True

if __name__ == '__main__':
    install_requirements()
    success = convert_md_to_pdf()
    if not success:
        sys.exit(1)
