
def getCenter(i):
    center = ((int(i[2]) + int(i[0])) // 2, (int(i[3]) + int(i[1])) // 2)
    return center


def distance(center1, center2):
    dist = ((int(center1[0]) - int(center2[0])) ** 2 + (int(center1[1]) - int(center2[1])) ** 2)**0.5
    return dist


def getHighRiskPeople(people_coordinates, SOCIAL_DISTANCE_THRESHOLD):
    highRiskPeople = []

    for i in range(0, len(people_coordinates)):
        center1 = getCenter(people_coordinates[i])
        for j in range(i + 1, len(people_coordinates)):
            center2 = getCenter(people_coordinates[j])

            dist = distance(center1, center2)
            if dist < SOCIAL_DISTANCE_THRESHOLD:
                if people_coordinates[i] not in highRiskPeople:
                    highRiskPeople.append(people_coordinates[i])

                if people_coordinates[j] not in highRiskPeople:
                    highRiskPeople.append(people_coordinates[j])

    lowRiskPeople = [people for people in people_coordinates if people not in highRiskPeople]

    return highRiskPeople, lowRiskPeople