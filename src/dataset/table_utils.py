from bs4 import BeautifulSoup


def extract_html_caption(html: str) -> str:
    """Return the text of the <caption> element in an HTML table, or empty string."""
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    if table:
        caption = table.find('caption')
        if caption:
            return caption.get_text(strip=True)
    return ''


def parse_html_table(html: str) -> list:
    """Parse an HTML table into a 2-D list of strings, expanding colspan/rowspan."""
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    if not table:
        return []

    rows = table.find_all('tr')
    grid: dict = {}
    max_cols = 0

    for r_idx, tr in enumerate(rows):
        c_idx = 0
        for cell in tr.find_all(['td', 'th']):
            while (r_idx, c_idx) in grid:
                c_idx += 1
            text = cell.get_text(separator=' ', strip=True)
            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))
            for dr in range(rowspan):
                for dc in range(colspan):
                    grid[(r_idx + dr, c_idx + dc)] = text
            c_idx += colspan
            max_cols = max(max_cols, c_idx)

    return [[grid.get((r, c), '') for c in range(max_cols)] for r in range(len(rows))]


def num_header_rows(html: str) -> int:
    """Return the number of header rows by inspecting the max rowspan in the first <tr>."""
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    if not table:
        return 1
    first_tr = table.find('tr')
    if not first_tr:
        return 1
    return max(int(c.get('rowspan', 1)) for c in first_tr.find_all(['td', 'th']))


def merge_header_rows(grid: list, n_header: int) -> list:
    """Collapse the first n_header rows into one by joining distinct cell values per column."""
    if n_header <= 1 or len(grid) < n_header:
        return grid
    merged_header = []
    for col_idx in range(len(grid[0])):
        seen = []
        for row_idx in range(n_header):
            val = grid[row_idx][col_idx].strip()
            if val and val not in seen:
                seen.append(val)
        merged_header.append(' '.join(seen))
    return [merged_header] + grid[n_header:]
