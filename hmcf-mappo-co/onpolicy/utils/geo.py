# -*- coding:utf-8 -*-

"""
地理信息库
"""
import math

PI = 3.1415926535897932
radian2degree = 180 / PI
degree2radian = PI / 180.0
NM2KM = 1.852  # 海里转千米

# WGS84 坐标系参数：
flattening = 1 / 298.257223563  # 扁率
EARTH_RADIUS_LONG = 6378137.0   # 地球长半轴 单位：米
EARTH_RADIUS_SHORT = EARTH_RADIUS_LONG - EARTH_RADIUS_LONG*flattening
# 6356752.3142451793
e_pow2 = 1 - (1 - flattening)*(1 - flattening) # 第一偏心率平方
EARTH_RADIUS = 6371137  # 地球平均半径


def get_horizontal_distance(geopoint1, geopoint2):
    """
     求地面两点的水平距离   Haversine公式
    :param geopoint1: tuple, (lat, lon), 例：(40.9, 140.0)
    :param geopoint2: tuple, (lat, lon), 例：(40.9, 142.0)
    :return: float, KM
    """
    lat1 = geopoint1[0] * degree2radian
    lon1 = geopoint1[1] * degree2radian
    lat2 = geopoint2[0] * degree2radian
    lon2 = geopoint2[1] * degree2radian

    difference = lat1 - lat2
    mdifference = lon1 - lon2
    distance = 2 * math.asin(math.sqrt(math.pow(math.sin(difference / 2), 2)
                                       + math.cos(lat1) * math.cos(lat2)
                                       * math.pow(math.sin(mdifference / 2), 2)))
    distance = distance * EARTH_RADIUS / 1000
    return distance


def get_slant_distance(geopoint1, geopoint2):
    """
    获取三维直线距离, 点高需为海拔高度
    :param geopoint1: tuple, (lat, lon, alt), 例：(40.9, 140.0, 560.8)
    :param geopoint2: tuple, (lat, lon, alt), 例：(40.9, 142.0, 4560.8)
    :return: float, KM
    """
    hd = get_horizontal_distance(geopoint1, geopoint2)
    delta_alt = geopoint1[2] - geopoint2[2]
    return get_range(hd, delta_alt)


def get_range(range_km, delta_alt):
    """
    获取直线距离
    :param range_km: float, 水平距离，KM
    :param delta_alt: float, 垂直距离，m
    :return: float, KM
    """
    range_km *= 1000.0
    return math.sqrt((range_km * range_km + delta_alt * delta_alt)) / 1000.0


def normal_angle(angle):
    """
    角度调整为0-360度以内
    :param angle: float, 角度
    :return: float
    """
    if 0 <= angle < 360:
        return angle
    else:
        return angle % 360


def get_azimuth(geopoint1, geopoint2):
    """
    获取point1 指向 point2 的方位角
    :param geopoint1: tuple, (lat, lon), 例：(40.9, 140.0)
    :param geopoint2: tuple, (lat, lon), 例：(40.9, 142.0)
    :return: 角度 0-360, 正北：0， 正东:90, 顺时针旋转，正西：270
    """
    lat1 = geopoint1[0] * degree2radian
    lon1 = geopoint1[1] * degree2radian
    lat2 = geopoint2[0] * degree2radian
    lon2 = geopoint2[1] * degree2radian
    azimuth = 180 * math.atan2(math.sin(lon2 - lon1), math.tan(lat2) * math.cos(lat1) - math.sin(lat1) * math.cos(lon2 - lon1)) / PI
    return normal_angle(azimuth)


def worldxyz2lla(x, y, z):
    """
    # WGS84 坐标系(x, y, z) 转 大地坐标系(lat, lon, alt)
    :param x: float, m
    :param y: float, m
    :param z: float, m
    :return: tuple, (lat, lon, alt), 单位：(degree,小数, 小数, m)
    """
    if not x:
        if not y:
            if z > 0:
                lon = 0.0
                lat = 90.0
                alt = z - EARTH_RADIUS_SHORT
            else:
                lon = 0.0
                lat = -90.0
                alt = -z - EARTH_RADIUS_SHORT
            return lat, lon, alt
        elif y > 0:
            lon = PI / 2
        else:
            lon = - PI/2
    else:
        lon = math.atan(math.fabs(y / x))
    if not z:
        # 在赤道平面，纬度为0
        lat = 0.0
        alt = math.sqrt(x * x + y * y) - EARTH_RADIUS_LONG
    else:
        lat = math.atan(math.fabs(z / math.sqrt(x * x + y * y)))
        for i in range(5):
            N = EARTH_RADIUS_LONG / math.sqrt(1.0 - e_pow2 * math.sin(lat) * math.sin(lat))
            alt = z / math.sin(lat) - N * (1.0 - e_pow2)
            lat = math.atan(z * (N + alt) / (math.sqrt(x * x + y * y) * (N * (1.0 - e_pow2) + alt)))
    if x < 0 and y > 0:
        lon = PI - lon
    elif x > 0 and y < 0:
        lon = -1.0 * lon
    elif x < 0 and y < 0:
        lon = lon - PI
    lat = lat * radian2degree
    lon = lon * radian2degree
    return lat, lon, alt


def lla2worldxyz(lat, lon, alt):
    """
    # 大地坐标系(lat, lon, alt) 转 WGS84 坐标系(x, y, z)
    :param lat: float, 纬度
    :param lon: float, 经度
    :param alt: float, 高度，m
    :return: x, y, z
    """
    N = EARTH_RADIUS_LONG / math.sqrt(1 - e_pow2 * math.sin(lat * degree2radian) * math.sin(lat * degree2radian))
    x = (N + alt) * math.cos(lat * degree2radian) * math.cos(lon * degree2radian)
    y = (N + alt) * math.cos(lat * degree2radian) * math.sin(lon * degree2radian)
    z = (N * (1 - e_pow2) + alt) * math.sin(lat * degree2radian)
    return x, y, z


def get_geopoint_from_distance(geo_point, azimuth, distance_m):
    """
    从地球海拔水平上，选一角度出发一定距离后，获取新的点. 距离越远，精度越差
    :param geo_point: tuple, (float, float), (纬度, 经度)
    :param azimuth: float, 角度，0-360， 正北为0， 顺时针旋转360度
    :param distance_m: float, 距离，单位：m
    :return: tuple, (lat, lon)
    """
    lat = geo_point[0]
    lon = geo_point[1]
    a = EARTH_RADIUS_LONG
    b = EARTH_RADIUS_SHORT
    alpha1 = azimuth * PI / 180
    sinAlpha1 = math.sin(alpha1)
    cosAlpha1 = math.cos(alpha1)

    tanU1 = (1 - flattening) * math.tan(lat*PI/180)
    cosU1 = 1 / math.sqrt((1 + tanU1*tanU1))
    sinU1 = tanU1 * cosU1
    sigma1 = math.atan2(tanU1, cosAlpha1)
    sinAlpha = cosU1 * sinAlpha1
    cosSqAlpha = 1 - sinAlpha * sinAlpha
    uSq = cosSqAlpha * (a*a - b*b) / (b*b)
    A = 1 + uSq/16384*(4096 + uSq*(-768 + uSq*(320 - 175*uSq)))
    B = uSq/1024*(256 + uSq*(-128 + uSq*(74 - 47*uSq)))

    sigma = distance_m / (b * A)
    sigmaP = 2 * PI
    sinSigma = 0
    cosSigma = 0
    for i in range(8):
        if math.fabs(sigma - sigmaP) < 1e-12:
            break
        cos2SigmaM = math.cos(2*sigma1 + sigma)
        sinSigma = math.sin(sigma)
        cosSigma = math.cos(sigma)
        deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma*(-1 + 2*cos2SigmaM*cos2SigmaM)
                                                                        - B/6*cos2SigmaM*(-3 + 4 * sinSigma*sinSigma) * (-3 + 4*cos2SigmaM*cos2SigmaM)))
        sigmaP = sigma
        sigma = distance_m / (b*A) + deltaSigma

    tmp = sinU1*sinSigma - cosU1*cosSigma*cosAlpha1
    lat2 = math.atan2(sinU1*cosSigma + cosU1*sinSigma*cosAlpha1,
                      (1 - flattening) * math.sqrt(sinAlpha*sinAlpha + tmp*tmp))
    lon_span = math.atan2(sinSigma*sinAlpha1, cosU1*cosSigma - sinU1*sinSigma*cosAlpha1)
    C = flattening / 16 * cosSqAlpha*(4 + flattening*(4 - 3*cosSqAlpha))
    lon_diff = lon_span - (1 - C)*flattening*sinAlpha*(sigma + C*sinSigma*(cos2SigmaM + C*cosSigma*(-1 + 2*cos2SigmaM*cos2SigmaM)))
    return lat2*180/PI, lon+lon_diff*180/PI


def get_two_point_dvalue(point_start, point_to, d_rate):
    """
    获取地球上两点的差值点
    :param point_start: tuple, (lat, lon), 例：(40.9, 140.0)
    :param point_to: tuple, (lat, lon), 例：(40.9, 142.0)
    :param d_rate: float, 差值比例, 0: 返回point_start, 1: 返回point_to, 2: 往到达点2倍距离点
    :return: tuple, (lat, lon)
    """
    if d_rate < 0:
        d_rate *= -1
        point_change = point_to
        point_to = point_start
        point_start = point_change

    distance = get_horizontal_distance(point_start, point_to) * d_rate * 1000
    azimuth = get_azimuth(point_start, point_to)
    return get_geopoint_from_distance(point_start, azimuth, distance)


def get_polygon_center(polygon):
    '''
    获取多边形的中心点
    :param polygon: list, 参考点列表,例:[(40, 39.0)]，其中纬度值在前，经度值在后
    :return: tuple, (lat, lon), 例：(40.9, 140.0)
    '''
    if polygon is None:
        return None
    
    n_count = len(polygon)
    if n_count == 0:
        return None
    
    lon_r = 0
    lat_r = 0
    for (lat, lon) in polygon:
        lat_r += lat
        lon_r += lon
        
    lon_r = lon_r/n_count
    lat_r = lat_r/n_count
    
    return (lat_r, lon_r)

# 速度单位转换
def speed_kmh2ms(speed_kmh):
    '''
    速度单位转换
    :param speed_kmh: float, km/h
    :return: float, m/s
    '''
    return speed_kmh * 1000 / 3600

def speed_ms2kmh(speed_ms):
    '''
    速度单位转换
    :param speed_ms: float, m/s
    :return: float, km/h
    '''
    return speed_ms * 3600 / 1000

#判断一个点是否在该多边形内
def IsPtInPoly(aLat, aLon, pointList):
    '''
    :param aLat: double 纬度
    :param aLon: double 经度
    
    :param pointList: list [(lat, lon)...] 多边形点的顺序需根据顺时针或逆时针，不能乱
    '''   
    iSum = 0
    iCount = len(pointList)
    
    if(iCount < 3):
        return False
        
    for i in range(iCount):
        pLon1 = pointList[i][1]
        pLat1 = pointList[i][0]    
        if(i == iCount - 1):
            pLon2 = pointList[0][1]
            pLat2 = pointList[0][0]
        else:
            pLon2 = pointList[i + 1][1]
            pLat2 = pointList[i + 1][0]
        
        if ((aLat >= pLat1) and (aLat < pLat2)) or ((aLat>=pLat2) and (aLat < pLat1)):
            if (abs(pLat1 - pLat2) > 0):
                pLon = pLon1 - ((pLon1 - pLon2) * (pLat1 - aLat)) / (pLat1 - pLat2)
                if(pLon < aLon):
                    iSum += 1
    if(iSum % 2 != 0):
        return True
    else:
        return False


if __name__ == "__main__":
    # 点1：(lat, lon, alt(单位:m))
    point1 = (0.0, 0.0, 0)
    # 点2：(lat, lon, alt(单位:m))
    point2 = (30.0, 42.6, 3000)
    point_list = [[31.2, 41.8, 375],
                    [31.2, 41.9, 958],
                    [31.2, 42.3, 306],
                    [30.8, 42.4, 1255],
                    [30.2, 42.7, 8412],
                    [30.1, 42.9, 467],
                    [30.0, 43.2, 412],
                    [29.6, 43.9, 10518],
                    [29.2, 44.4, 528],
                    [29.0, 45.9, 8563],
                    [29.0, 46.5, 2536]]

    # 求水平距离
    d1 = get_horizontal_distance(point2, point_list[0])
    print('{},{} horizontal distance:{}KM'.format(point2, point_list[0], d1))

    # 求斜面距离
    d1 = get_slant_distance(point2, point_list[0])
    print('{},{} distance:{}KM'.format(point2, point_list[0], d1))

    # 求点1往东10km的点
    east_point = get_geopoint_from_distance(point1, 90, 10000)
    print("{} east 10km is {}".format(point1, east_point))

    # 求点1往西北200km的点
    wn_point = get_geopoint_from_distance(point1, 315, 200000)
    print("{} northwest 200km is {}".format(point1, wn_point))

    # 点1到点2指向角度 正北为0，顺时针旋转到360
    azimuth = get_azimuth(point1, point2)
    print("{} to {}, azimuth degree {} degree".format(point1, point2, azimuth))

    # 点2到点3分成3份，求中间两个点
    th_point1 = get_two_point_dvalue(point2, point_list[0], 1/3)
    th_point2 = get_two_point_dvalue(point2, point_list[0], 2/3)
    print("{} and {} Divide into three parts is {} and {}".format(point2, point_list[0], th_point1, th_point2))
