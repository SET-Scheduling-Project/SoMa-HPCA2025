#include "utils.h"

#include <cassert>
#include <iostream>
#include <algorithm>
#include <numeric>
vol_t ofm_ubuf_vol;
vol_t core_buffer_read_bandwidth;

cost_t default_cost(energy_t energy, cycle_t time)
{
	return energy * time;
}

std::function<cost_t(energy_t, cycle_t)> cost_func = default_cost;

// Used for better io of int8 and uint8
std::istream &operator>>(std::istream &in, std::int8_t &num)
{
	std::int16_t n;
	in >> n;
	num = static_cast<std::int8_t>(n);
	return in;
}

std::ostream &operator<<(std::ostream &out, const std::int8_t &num)
{
	return out << +num;
}

std::istream &operator>>(std::istream &in, std::uint8_t &num)
{
	std::uint16_t n;
	in >> n;
	num = static_cast<std::uint8_t>(n);
	return in;
}

std::ostream &operator<<(std::ostream &out, const std::uint8_t &num)
{
	return out << +num;
}

cost_t calc_cost(energy_t energy, cycle_t time)
{
	if (energy >= energy_inf)
	{
		return cost_inf;
	}
	return cost_func(energy, time);
}

len_t *part_intv(len_t tot_len, len_t ncuts)
{
	len_t *arr = new len_t[ncuts + 1];
	for (len_t i = 0; i <= ncuts; ++i)
	{
		arr[i] = tot_len - (tot_len * (ncuts - i)) / ncuts;
	}
	return arr;
}

bool pos_t::operator<(const pos_t &other) const
{
	return (x < other.x) || ((x == other.x) && (y < other.y));
}

bool pos_t::operator==(const pos_t &other) const
{
	return x == other.x && y == other.y;
}

bool pos_t::operator>(const pos_t &other) const
{
	return (x > other.x) || ((x == other.x) && (y > other.y));
}

bool pos_t::operator<=(const pos_t &other) const
{
	return (x < other.x) || ((x == other.x) && (y <= other.y));
}

bool pos_t::operator>=(const pos_t &other) const
{
	return (x > other.x) || ((x == other.x) && (y >= other.y));
}

bool pos_t::operator!=(const pos_t &other) const
{
	return x != other.x || y != other.y;
}

std::ostream &operator<<(std::ostream &os, const pos_t &pos)
{
	return os << '(' << static_cast<int>(pos.x) << ',' << static_cast<int>(pos.y) << ')';
}

tensor_shape::tensor_shape(len_t _bk, len_t _c, len_t _h, len_t _w)
	: bk(_bk), c(_c), h(_h), w(_w == 0 ? _h : _w)
{
	update_size();
}

void tensor_shape::update_size()
{
	size = bk * c * h * w;
	// assert(size > 0);
}

bool tensor_shape::operator==(const tensor_shape &other) const
{
	return bk == other.bk && c == other.c && h == other.h && w == other.w;
	// TODO: for speedup.
	// return memcmp(this, &other, sizeof(other));
}

vol_t tensor_shape::get_size() const
{
	return size;
}

std::ostream &operator<<(std::ostream &os, const tensor_shape &shape)
{
	return os << "(B/K=" << shape.bk << ", C=" << shape.c << ", H=" << shape.h << ", W=" << shape.w << ')';
}

fmap_shape::fmap_shape(len_t _c, len_t _h, len_t _w)
	: c(_c), h(_h), w(_w == 0 ? _h : _w)
{
	update_size();
}

bool fmap_shape::operator==(const fmap_shape &other) const
{
	return c == other.c && h == other.h && w == other.w;
	// TODO: for speedup.
	// return memcmp(this, &other, sizeof(other));
}

void fmap_shape::update_size()
{
	size = c * h * w;
	// assert(size > 0);
}

vol_t fmap_shape::tot_size(len_t batch_size) const
{
	return size * batch_size;
}

std::ostream &operator<<(std::ostream &os, const fmap_shape &shape)
{
	return os << "(C=" << shape.c << ", H=" << shape.h << ", W=" << shape.w << ')';
}

bool fmap_range::dim_range::operator<(const dim_range &other) const
{
	if (from != other.from)
		return from < other.from;
	return to < other.to;
}

bool fmap_range::dim_range::operator==(const dim_range &other) const
{
	return from == other.from && to == other.to;
}

bool fmap_range::dim_range::operator!=(const dim_range &other) const
{
	return from != other.from || to != other.to;
}

std::ostream &operator<<(std::ostream &os, const fmap_range::dim_range &range)
{
	return os << '(' << range.from << ',' << range.to << ')';
}

fmap_range::dim_range &fmap_range::dim_range::operator+=(const len_t &offset)
{
	from += offset;
	to += offset;
	return *this;
}

fmap_range::dim_range &fmap_range::dim_range::operator-=(const len_t &offset)
{
	from -= offset;
	to -= offset;
	return *this;
}

bool fmap_range::dim_range::is_empty() const
{
	return to <= from;
}

vol_t fmap_range::dim_range::size() const
{
	return to - from;
}

fmap_range::dim_range fmap_range::dim_range::intersect(const dim_range &other) const
{
	len_t newFrom = MAX(from, other.from);
	len_t newTo = MIN(to, other.to);
	return dim_range{newFrom, MAX(newTo, newFrom)};
}

fmap_range::dim_range fmap_range::dim_range::combine(const dim_range& other) const{
	len_t newFrom = MIN(from, other.from);
	len_t newTo = MAX(to, other.to);
	return dim_range{newFrom, MAX(newTo, newFrom)};
}

fmap_range::fmap_range(const fmap_shape &shape, len_t B)
	: c{0, shape.c}, b{0, B}, h{0, shape.h}, w{0, shape.w} {}

const fmap_range::dim_range &fmap_range::get_range(uint8_t idx) const
{
	switch (idx)
	{
	case 0:
		return c;
	case 1:
		return b;
	case 2:
		return h;
	case 3:
		return w;
	default:
		assert(false);
	}
	return c;
}

fmap_range::fmap_range(const dim_range &_c, const dim_range &_b, const dim_range &_h, const dim_range &_w)
	: c(_c), b(_b), h(_h), w(_w) {}

vol_t fmap_range::size() const
{
	return c.size() * b.size() * h.size() * w.size();
}

bool fmap_range::operator<(const fmap_range &other) const
{
	if (c != other.c)
		return c < other.c;
	if (b != other.b)
		return b < other.b;
	if (h != other.h)
		return h < other.h;
	return w < other.w;
}

bool fmap_range::operator==(const fmap_range &other) const
{
	return c == other.c && b == other.b && h == other.h && w == other.w;
}

fmap_range fmap_range::intersect(const fmap_range &other) const
{
	return fmap_range{
		c.intersect(other.c),
		b.intersect(other.b),
		h.intersect(other.h),
		w.intersect(other.w)};
}

fmap_range fmap_range::combine(const fmap_range& other) const{
	return fmap_range{
		c.combine(other.c),
		b.combine(other.b),
		h.combine(other.h),
		w.combine(other.w)
	};
}

bool fmap_range::is_empty() const
{
	return c.is_empty() || b.is_empty() || h.is_empty() || w.is_empty();
}

std::ostream &operator<<(std::ostream &os, const fmap_range &range)
{
	return os << "(B=" << range.b << ", C=" << range.c << ", H=" << range.h << ", W=" << range.w << ')';
}

tensor_shape fmap_range_to_tensor_shape(const fmap_range& range)
{
	return tensor_shape(range.b.size(), range.c.size(), range.h.size(), range.w.size());
}

fmap_range tensor_shape_to_fmap_range(const tensor_shape& shape)
{
	return fmap_range(fmap_shape(shape.c, shape.h, shape.w), shape.bk);
}

cidx_t dis(const pos_t &x, const pos_t &y)
{
	return std::abs(x.x - y.x) + std::abs(x.y - y.y);
}
len_t getGCD(len_t a, len_t b)
{
#ifdef __GNUC__
	return std::__gcd(a, b);
#else
	return std::gcd(a, b);
#endif
}
len_t getLCM(len_t a, len_t b)
{
	return std::lcm(a, b);
}
int find_root(int *fa, int node) {
    if(fa[node] != node)
    	fa[node] = find_root(fa, fa[node]);
    return fa[node];
}
void unity(int *fa, int x, int y)
{
    int r1 = find_root(fa, x); 
    int r2 = find_root(fa, y); 
    fa[r1] = r2;
}
/*
int divceil(int m, int n) {
	return (m - 1) / n + 1;
}
*/
std::tuple<int, int> getSegment(int x, int y, int i) {
    if (y <= 0 || i < 1 || i > y) {
        throw std::invalid_argument("Invalid y or i");
    }
    
    int baseLength = x / y;
    int extra = x % y;
    
    int start = (i - 1) * baseLength + std::min(i - 1, extra);
    int end = start + baseLength + (i <= extra ? 1 : 0) - 1;
    
    return std::make_tuple(start, end);
}

std::random_device::result_type _seed_here = 0;
std::mt19937 gen(_seed_here);

void set_gen_seed() {
	gen.seed(_seed_here);
}