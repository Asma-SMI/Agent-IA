from reportlab.pdfgen import canvas

def make_pdf(path="test_docucourtments.pdf"):
    c = canvas.Canvas(path)
    c.setFont("Helvetica", 14)

    c.drawString(50, 800, "RECU BANCAIRE")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, "Banque: BIAT")
    c.drawString(50, 750, "Date: 18/11/2025")
    c.drawString(50, 730, "Code titre: XYZ-2025-11")
    c.drawString(50, 710, "Montant: 1250.75 TND")
    c.drawString(50, 690, "Reference: A4587")

    c.showPage()

    c.setFont("Helvetica", 14)
    c.drawString(50, 800, "PAGE 2 - Informations supplementaires")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, "Nom client: Ali Ben Salah")

    c.save()
    print("PDF créé:", path)

if __name__ == "__main__":
    make_pdf()
