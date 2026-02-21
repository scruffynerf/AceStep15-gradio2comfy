import struct

def create_vorbis_comment_block(comment_dict, last_block):
    """Create FLAC vorbis comment metadata block"""
    vendor_string = b'ComfyUI-ACE-Step'
    vendor_length = len(vendor_string)

    comments = []
    for key, value in comment_dict.items():
        comment = f"{key}={value}".encode('utf-8')
        comments.append(struct.pack('<I', len(comment)) + comment)

    num_comments = len(comments)
    comments_data = b''.join(comments)

    block_data = (
        struct.pack('<I', vendor_length) + vendor_string +
        struct.pack('<I', num_comments) + comments_data
    )

    block_header = struct.pack('>I', len(block_data))[1:]
    if last_block:
        block_header = bytes([block_header[0] | 0x80]) + block_header[1:]
    block_header = bytes([4]) + block_header

    return block_header + block_data
