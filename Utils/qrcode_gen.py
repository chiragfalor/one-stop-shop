import qrcode

# make a function to generate a QR code

def make_qr(data):
    img = qrcode.QRCode(box_size=100,
                        border=2,
                        image_factory=None,
                        mask_pattern=None,
                        error_correction=qrcode.constants.ERROR_CORRECT_L,)
    img.add_data(data)
    img.make(fit=True)
    img = img.make_image(fill_color='black', back_color='white')
    return img

if __name__ == '__main__':
    # make a QR code
    text = "chiragfalor"
    save_name = "qr"
    img = make_qr(text)
    img.save(f'QRCodes/{save_name}.png')


