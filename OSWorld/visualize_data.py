import os
import json
from fpdf import FPDF
from PIL import Image
import base64
import io

json_path = "result after tosft.py.json" 
save_pdf_path = "sft_fail.pdf"

pdf = FPDF(orientation="P", unit="pt", format="A4")
pdf.set_auto_page_break(auto=True, margin=20)

font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux defaut path
pdf.add_font("DejaVu", "", font_path, uni=True)
pdf.add_font("DejaVu", "B", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", uni=True) 
pdf.set_font("DejaVu", size=10)

page_width = 595
margin = 20
spacing = 10
images_per_row = 2
max_width = (page_width - 2 * margin - (images_per_row - 1) * spacing) // images_per_row

# 加载 JSON
with open(json_path, "r") as f:
    data = json.load(f)
import random
for task_id, task in enumerate(random.sample(data, 20)):
    pdf.add_page()
    pdf.set_font("DejaVu", size=10)
    pdf.multi_cell(0, 14, f"Task #{task_id + 1}", align="L")
    pdf.ln(6)

    for step_id, step in enumerate(task):
        try:
            role = step['role']
        except:
            import pdb
            pdb.set_trace()
        content = step['content'][0]

        if content['type'] == "text":
            text = content['text']
            pdf.set_font("DejaVu", style='B' if role == 'user' else '', size=9)
            prefix = "User: " if role == 'user' else "Assistant: "
            pdf.multi_cell(0, 12, prefix + text, align="L")
            pdf.ln(2)

        elif content['type'] == "image_url":
            img_data = content['image_url']['url']
            if img_data.startswith("data:image/png;base64,"):
                img_base64 = img_data.split(",")[1]
                img_bytes = base64.b64decode(img_base64)
                img = Image.open(io.BytesIO(img_bytes))
                print(img.size)
                w, h = img.size
                scale = max_width / w
                new_w = max_width
                new_h = h * scale

                temp_img_path = f"temp_{task_id}_{step_id}.png"
                img.save(temp_img_path)

                x = margin
                y = pdf.get_y()
                pdf.image(temp_img_path, x=x, y=y, w=new_w)
                pdf.set_y(y + new_h + spacing)

                os.remove(temp_img_path)

pdf.output(save_pdf_path)
print(f"PDF saved to: {save_pdf_path}")
